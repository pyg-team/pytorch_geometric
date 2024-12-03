import sys

import torch

import torch_geometric
from torch_geometric.contrib.nn import HIN2Vec
from torch_geometric.datasets import DBLP

dataset = DBLP(root='data/DBLP')
data = dataset[0]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

model = HIN2Vec(data.edge_index_dict, embedding_dim=128, metapath_length=3,
                walk_length=100, walks_per_node=1, num_negative_samples=1,
                reg='sigmoid', sparse=True).to(device)

num_workers = 4 if sys.platform == 'linux' else 0
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  f'Loss: {total_loss / log_steps:.4f}')
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  f'Acc: {acc:.4f}')


@torch.no_grad()
def test():
    model.eval()

    z, _ = model('author')
    y = data['author'].y
    train_mask = data['author'].train_mask
    test_mask = data['author'].test_mask

    accuracy = model.test(
        train_z=z[train_mask],
        train_y=y[train_mask],
        test_z=z[test_mask],
        test_y=y[test_mask],
        max_iter=150,
    )
    return accuracy


for epoch in range(1, 26):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
