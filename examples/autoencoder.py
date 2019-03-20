import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(Encoder(dataset.num_features, out_channels=16)).to(device)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = model.split_edges(data)
x, edge_index = data.x.to(device), data.edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    loss = model.loss(z, data.train_pos_edge_index, data.train_neg_adj_mask)
    loss.backward()
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.encoder.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return model.eval(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 201):
    train()
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
print('Test AUC: {:.4f}, Test AP: {:.4f}'.format(auc, ap))
