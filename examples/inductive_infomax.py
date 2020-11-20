import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv, DeepGraphInfomax
'''
In '"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>' two methods
were provided, a transductive method for small graphs (see examples/infomax.py)
and a inductive method for large graphs, implemented here. The inductive implementation
utilises the mean pooling propagation rule as detailed in `"Inductive Representation 
Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`.
'''

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[10,10,25], batch_size=256, shuffle=True,
                               num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=256, shuffle=False,
                                  num_workers=12)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        

        convs = []
        convs.append(SAGEConv(in_channels, hidden_channels))
        convs.append(nn.PReLU(hidden_channels))
        convs.append(SAGEConv(hidden_channels, hidden_channels))
        convs.append(nn.PReLU(hidden_channels))
        convs.append(SAGEConv(hidden_channels, hidden_channels))
        convs.append(nn.PReLU(hidden_channels))

        self.convs = nn.Sequential(*convs)

    def forward(self, x, edge_index):

        return self.convs(x)


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
