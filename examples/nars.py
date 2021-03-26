import time
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, ReLU, Dropout, ModuleList
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.nn import NARS

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OGB')
dataset = PygNodePropPredDataset('ogbn-mag', root)
split_idx = dataset.get_idx_split()
data = dataset[0]
print(data)

train_idx = split_idx['train']['paper']
val_idx = split_idx['valid']['paper']
test_idx = split_idx['test']['paper']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Let's execute NARS pre-processing on GPU for some speed-up.
data = data.to(device)

# We will fill missing features with zeros for now.
# Performance can be drastically increased by using TransE embeddings.
for key in set(data.num_nodes_dict.keys()).difference(set(['paper'])):
    data.x_dict[key] = torch.zeros(data.num_nodes_dict[key], 128,
                                   device=device)

nars = NARS(num_hops=5, num_sampled_subsets=8, num_features=128).to(device)

t = time.perf_counter()
print('Pre-processing features...', end=' ', flush=True)
aggr_dict = nars(data.x_dict, data.edge_index_dict)
print(f'Done! [{time.perf_counter() - t:.2f}s]')

# We only care about paper embeddings...
x = aggr_dict['paper']  # [num_hops, num_subsets, num_nodes, num_features]
y = data.y_dict['paper'].view(-1)  # [num_nodes]
del aggr_dict
del data


class SIGN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_hops,
                 dropout=0.5):
        super(SIGN, self).__init__()

        self.mlps = ModuleList()
        for _ in range(num_hops):
            mlp = Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.mlps.append(mlp)

        self.mlp = Sequential(
            Dropout(dropout),
            Linear(num_hops * hidden_channels, hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, xs):
        outs = [mlp(x) for x, mlp in zip(xs, self.mlps)]
        return self.mlp(torch.cat(outs, dim=-1))


sign = SIGN(x.size(-1), 512, dataset.num_classes, num_hops=6).to(device)
optimizer = torch.optim.Adam(
    list(nars.parameters()) + list(sign.parameters()), lr=0.001)


def train(x, y, idx):
    nars.train()
    sign.train()

    total_loss = num_examples = 0
    for idx in DataLoader(idx, batch_size=50000, shuffle=True):
        x_mini = x[:, :, idx].to(device)
        y_mini = y[idx].to(device)

        optimizer.zero_grad()
        x_mini = nars.weighted_aggregation(x_mini)
        out = sign(x_mini.unbind(0))
        loss = F.cross_entropy(out, y_mini)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
        num_examples += idx.numel()

    return total_loss / num_examples


@torch.no_grad()
def test(x, y, idx):
    total_correct = num_examples = 0
    for idx in DataLoader(idx, batch_size=100000):
        x_mini = x[:, :, idx].to(device)
        y_mini = y[idx].to(device)

        x_mini = nars.weighted_aggregation(x_mini)
        pred = sign(x_mini.unbind(0)).argmax(dim=-1)

        total_correct += int((pred == y_mini).sum())
        num_examples += idx.numel()

    return total_correct / num_examples


for epoch in range(1, 201):
    loss = train(x, y, train_idx)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 10 == 0:
        train_acc = test(x, y, train_idx)
        val_acc = test(x, y, val_idx)
        test_acc = test(x, y, test_idx)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')
