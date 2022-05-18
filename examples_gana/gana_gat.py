import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from torch_geometric.nn import GATConv
from debug import debug

import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=2, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=2, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI_GANA')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(train_dataset.num_features, hidden_channels, heads=8, dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*8, hidden_channels*8,
                 heads=1, concat=True,  dropout=0.6))
        self.convs.append(GATConv(hidden_channels*8, train_dataset.num_classes,
         heads=1, concat=False,dropout=0.6))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        for conv in self.convs:
            x1 = conv(x, edge_index)
            x = F.elu(x1)
            x = F.dropout(x, p=0.6,training=self.training)
        return x1 #F.log_softmax(x1, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # target = torch.tensor([row.index(1) for row in data.y.tolist()]).to(device)
        loss =loss_op(model(data.x, data.edge_index), data.y)
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



for epoch in range(1, int(args.epochs)):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))



import os
result_dir = f"results/{os.path.basename(__file__).split('.')[0]}/{args.num_layers}_{args.hidden_channels}"
print(result_dir)
os.makedirs(result_dir, exist_ok=True)
debug(val_loader, result_dir, model)
