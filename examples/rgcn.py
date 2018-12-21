import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv

name = 'MUTAG'
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
dataset = Entities(path, name)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(
            data.num_nodes, 16, dataset.num_relations, num_bases=30)
        self.conv2 = RGCNConv(
            16, dataset.num_classes, dataset.num_relations, num_bases=30)

    def forward(self, edge_index, edge_type, edge_norm):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    F.nll_loss(out[data.train_idx], data.train_y).backward()
    optimizer.step()


def test():
    model.eval()
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = out[data.test_idx].max(1)[1]
    acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    return acc


for epoch in range(1, 51):
    train()
    test_acc = test()
    print('Epoch: {:02d}, Accuracy: {:.4f}'.format(epoch, test_acc))
