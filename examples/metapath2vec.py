import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'AMiner')
dataset = AMiner(path)
data = dataset[0]
print(data)
N = data.num_nodes_dict['author']
loader = DataLoader(torch.arange(N), batch_size=128, shuffle=True)

metapath = [
    ('author', 'wrote', 'paper'),
    ('paper', 'published in', 'venue'),
    ('venue', 'published', 'paper'),
    ('paper', 'written by', 'author'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                     metapath=metapath, walks_per_node=10, sparse=True)
model = model.to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)


def train(epoch, log_steps=500):
    model.train()

    total_loss = 0
    for i, subset in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(subset)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0


@torch.no_grad()
def test():
    model.eval()

    z = model('author', subset=data.y_index_dict['author'])
    y = data.y_dict['author']

    # 10/90 Split:
    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * 0.1)]
    test_perm = perm[int(z.size(0) * 0.1):]

    return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                      max_iter=150)


for epoch in range(1, 6):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
