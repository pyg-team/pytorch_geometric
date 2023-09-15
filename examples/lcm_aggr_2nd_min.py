'''final validation accuracy: 95%'''
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch_geometric.nn as gnn
from torch_geometric.nn.aggr import LCMAggregation

seed = 42
num_bits = 8
emb_dim = 128
batch_size = 32
num_epochs = 1000
opt_lr = 1e-4
p_dropout = .25
trainset_size = 2**16
validset_size = 2**10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(seed)


class Random2ndMinimumDataset(Dataset):
    '''A labeled dataset, where each sample is a multiset of integers
    encoded as bit-vectors, and the label is the second smallest integer
    in the multiset.
    '''
    def __init__(self, dataset_sz, num_bits, multiset_sz, permute=False):
        self.dataset_sz = dataset_sz
        self.num_bits = num_bits
        self.multiset_sz = multiset_sz
        self.permute = permute
        self.generate_dataset()

    def generate_dataset(self):
        self.dataset = []
        for _ in range(self.dataset_sz):
            self.dataset += [self.generate_sample()]

    def generate_sample(self):
        if isinstance(self.multiset_sz, tuple):
            # randomly sample multiset size
            sz = torch.randint(*self.multiset_sz, (1, ))
        elif isinstance(self.multiset_sz, int):
            sz = self.multiset_sz
        else:
            raise ValueError("`multiset_sz` must be a tuple or an int")

        # randomly sample a multiset of integers of size `sz`, encoded as
        # bit-vectors
        bitvecs = torch.randint(0, 2, (sz, num_bits))

        # convert bit-vectors to integers
        ints = self.bitvecs_to_ints(bitvecs)

        # find the second smallest element in the multiset, which is the target
        target_idx = torch.topk(ints, 2, largest=False).indices[-1]
        target = bitvecs[target_idx].float()

        return bitvecs, target

    def bitvecs_to_ints(self, bitvecs):
        powers = torch.pow(2, torch.arange(num_bits).flip(0)).unsqueeze(0)
        return (bitvecs * powers).sum(-1)

    def __getitem__(self, i):
        multiset, target = self.dataset[i]
        if self.permute:
            multiset = multiset[torch.randperm(multiset.size(0))]
        return multiset, target

    def __len__(self):
        return len(self.dataset)


def collate_fn(samples):
    x, y, index = tuple(
        zip(*[(multiset, target, torch.full((multiset.size(0), ), i))
              for i, (multiset, target) in enumerate(samples)]))
    x = torch.cat(x)
    y = torch.stack(y)
    index = torch.cat(index)
    return x, y, index


# train on multisets of random sizes between 2 and 16
trainset = Random2ndMinimumDataset(trainset_size, num_bits, (2, 16 + 1),
                                   permute=True)
train_dl = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn,
                      shuffle=True)

# validate on multisets of size 32, larger than in training
validset = Random2ndMinimumDataset(validset_size, num_bits, 32)
valid_dl = DataLoader(validset, batch_size=len(validset),
                      collate_fn=collate_fn)

print('Generated train and validation datasets.')


class BitwiseEmbedding(nn.Module):
    def __init__(self, num_bits, emb_dim):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(2, emb_dim) for _ in range(num_bits)])

    def forward(self, bitvecs):
        bit_embeddings = [emb(b) for emb, b in zip(self.embs, bitvecs.T)]
        return torch.stack(bit_embeddings).sum(0)


enc = nn.Sequential(BitwiseEmbedding(num_bits, emb_dim),
                    nn.Linear(emb_dim, emb_dim), nn.Dropout(p_dropout),
                    nn.GELU())

agg = LCMAggregation(emb_dim, emb_dim, project=False)

dec = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(p_dropout),
                    nn.GELU(), nn.Linear(emb_dim, num_bits))

net = gnn.Sequential('x, index', [(enc, 'x -> x'), (agg, 'x, index -> x'),
                                  (dec, 'x -> x')])
net.to(device)

criterion = nn.BCEWithLogitsLoss(reduction='none')
opt = torch.optim.Adam(params=net.parameters(), lr=opt_lr)

print("Created model, optimizer, and loss function.")

print('Beginning training. Device:', device)


def train():
    losses = []
    for x, y, index in train_dl:
        x = x.to(device)
        y = y.to(device)
        index = index.to(device)

        pred = net(x, index)
        loss = criterion(pred, y).sum(-1).mean()
        losses += [loss.item()]

        opt.zero_grad()
        loss.backward()
        opt.step()
    losses = np.array(losses)
    return losses.mean(), losses.std()


@torch.no_grad()
def validate():
    total_correct = total_examples = 0
    for x, y, index in valid_dl:
        x = x.to(device)
        y = y.to(device)
        index = index.to(device)

        pred = torch.sigmoid(net(x, index)).round().squeeze()
        num_mistakes = (pred != y).sum(-1)
        total_correct += (num_mistakes == 0).sum()
        total_examples += y.size(0)
    return total_correct / total_examples


for ep in range(num_epochs):
    loss_mean, loss_std = train()
    val_acc = validate()
    print(f'ep {ep+1:04d}: losses=[{loss_mean:.4f}, {loss_std:.4f}],'
          f'acc={val_acc*100:.2f}%')
