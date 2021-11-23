import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import FAUST
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SplineConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FAUST')
pre_transform = T.Compose([T.FaceToEdge(), T.Constant(value=1)])
train_dataset = FAUST(path, True, T.Cartesian(), pre_transform)
test_dataset = FAUST(path, False, T.Cartesian(), pre_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5, aggr='add')
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5, aggr='add')
        self.lin1 = torch.nn.Linear(64, 256)
        self.lin2 = torch.nn.Linear(256, d.num_nodes)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        optimizer.zero_grad()
        F.nll_loss(model(data.to(device)), target).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes)


for epoch in range(1, 101):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
