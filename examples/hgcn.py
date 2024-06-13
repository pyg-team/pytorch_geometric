import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import HGCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'HGCN-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


class LinearDecoder(torch.nn.Module):
    """MLP Decoder for Hyperbolic/Euclidean node classification models.
    """
    def __init__(self, manifold, in_dim, out_dim, c):
        super(LinearDecoder, self).__init__()
        self.manifold = manifold
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.cls = torch.nn.Linear(self.input_dim, self.output_dim)
        self.c = torch.nn.Parameter(torch.FloatTensor([c]))

    def forward(self, x):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c),
                                    c=self.c)
        x = self.cls(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.input_dim, self.output_dim, self.c)


class HGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HGCNConv(in_channels, hidden_channels, dropout=0.5)
        self.decoder = LinearDecoder(self.conv1.manifold, hidden_channels,
                                     out_channels, 1.0)

    def forward(self, x, edge_index):
        # print(x.shape, edge_index.shape)
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.decoder(x)
        x = F.log_softmax(x, dim=-1)
        # print(x.shape)
        # exit(0)
        return x


model = HGCN(dataset.num_features, args.hidden_channels,
             dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
print(model)


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


times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
