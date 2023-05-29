import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import DirSageConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chameleon')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'Dir-Sage-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'WikipediaNetwork')
dataset = WikipediaNetwork(root=path, name=args.dataset,
                           transform=T.NormalizeFeatures())
data = dataset[0]


class DirSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.5):
        super().__init__()
        self.conv1 = DirSageConv(in_channels, hidden_channels, alpha=alpha)
        self.conv2 = DirSageConv(hidden_channels, out_channels, alpha=alpha)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = DirSage(dataset.num_features, args.hidden_channels,
                dataset.num_classes, alpha=1)
model, data = model.to(device), data.to(device)
data.train_mask, data.val_mask, data.test_mask = data.train_mask[:,
                                                                 0], data.val_mask[:,
                                                                                   0], data.test_mask[:,
                                                                                                      0]
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
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
