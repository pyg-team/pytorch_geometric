import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.nn import DirGNNConv, GCNConv, SAGEConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chameleon')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--conv', type=str, default='gcn')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Wikipedia')
dataset = WikipediaNetwork(
    root=path,
    name=args.dataset,
    transform=T.NormalizeFeatures(),
)

data = dataset[0].to(device)
data.train_mask = data.train_mask[:, 0]
data.val_mask = data.val_mask[:, 0]
data.test_mask = data.test_mask[:, 0]

if args.conv == 'gcn':
    Conv = GCNConv
elif args.conv == 'sage':
    Conv = SAGEConv
else:
    raise NotImplementedError


class DirGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha):
        super().__init__()
        self.conv1 = Conv(in_channels, hidden_channels)
        self.conv1 = DirGNNConv(self.conv1, alpha, root_weight=False)

        self.conv2 = Conv(hidden_channels, out_channels)
        self.conv2 = DirGNNConv(self.conv2, alpha, root_weight=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = DirGNN(
    dataset.num_features,
    args.hidden_channels,
    dataset.num_classes,
    alpha=args.alpha,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
