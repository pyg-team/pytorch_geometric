import os.path as osp
import argparse
import csv
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import PPI
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=2, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=2, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI_GANA')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# if args.use_gdc:
#     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
#                 normalization_out='col',
#                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
#                 sparsification_kwargs=dict(method='topk', k=128,
#                                            dim=0), exact=True)
#     data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(train_dataset.num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, train_dataset.num_classes))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x1 = conv(x, edge_index)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


@torch.no_grad()
def debug(loader):
    model.eval()
    merged = []
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())
        debug = torch.cat((data.x.to(device), data.y.to(device), (out > 0)), dim=1)
        merged.append(debug.cpu())
    with open("debug_sage_result.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(merged)

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

for epoch in range(1, int(args.epochs)):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
# debug(test_loader)
