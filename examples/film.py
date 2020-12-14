import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, FiLMConv
import torch_geometric.transforms as T
from sklearn.metrics import f1_score

path = osp.join("/home/laurent/graph_data", 'PPI')
train_dataset = PPI(path, split='train')
val_dataset   = PPI(path, split='val')
test_dataset  = PPI(path, split='test')
# you might want to add self-loops as follows
'''
train_dataset = PPI(path, transform=T.AddSelfLoops(), split='train')
val_dataset   = PPI(path, transform=T.AddSelfLoops(), split='val')
test_dataset  = PPI(path, transform=T.AddSelfLoops(), split='test')
'''
train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=2, shuffle=False)
test_loader   = DataLoader(test_dataset,  batch_size=2, shuffle=False)


num_hidden = 256

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = FiLMConv(train_dataset.num_features, num_hidden,
                              nn=Linear(train_dataset.num_features, 2*num_hidden),
                              bias=False)
        self.lin1 = Linear(train_dataset.num_features, num_hidden)
        self.conv2 = FiLMConv(num_hidden, num_hidden,
                              nn=Linear(num_hidden, 2*num_hidden),
                              bias=False)
        self.lin2 = Linear(num_hidden, num_hidden)
        self.conv3 = FiLMConv(num_hidden, train_dataset.num_classes,
                              nn=Linear(num_hidden, 2*train_dataset.num_classes),
                              bias=False)
        self.lin3 = Linear(num_hidden, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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


for epoch in range(1, 101):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
