import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool_x,
                                global_mean_pool)

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_geometric.nn.pool.max_pool import _max_pool_x

from typing import Optional

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
transform = T.Cartesian(cat=False)
train_dataset = MNISTSuperpixels(path, True, transform=transform)
test_dataset = MNISTSuperpixels(path, False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
d = train_dataset


def cartesian_jit(x: Optional[torch.Tensor],
                  edge_index: Optional[torch.Tensor],
                  edge_attr: Optional[torch.Tensor],
                  pos: Optional[torch.Tensor],
                  batch: Optional[torch.Tensor],
                  cat: bool = False,
                  norm: bool = True,
                  max: Optional[float] = None):
    assert pos is not None
    assert edge_index is not None

    row, col, pos, pseudo = edge_index[0], edge_index[1], pos, edge_attr

    cart = pos[col] - pos[row]
    cart = cart.view(-1, 1) if cart.dim() == 1 else cart

    if norm and cart.numel() > 0:
        max_value = cart.abs().max() if max is None else torch.tensor(max)
        cart = cart / (2 * max_value) + 0.5

    if pseudo is not None and cat:
        pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
        edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
    else:
        edge_attr = cart

    return x, edge_index, edge_attr, pos, batch


def max_pool_jit(cluster,
                 x,
                 edge_index,
                 edge_attr: Optional[torch.Tensor],
                 pos,
                 batch: Optional[torch.Tensor]):
    cluster, perm = consecutive_cluster(cluster)

    x = None if x is None else _max_pool_x(cluster, x)
    edge_index, attr = pool_edge(cluster, edge_index, edge_attr)
    batch = None if batch is None else pool_batch(perm, batch)
    pos = None if pos is None else pool_pos(cluster, pos)

    x, edge_index, edge_attr, pos, batch = \
        cartesian_jit(x, edge_index, edge_attr, pos, batch)

    return x, edge_index, edge_attr, pos, batch


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index[0], edge_index[1]
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


init_data = None
for i, tdata in enumerate(test_loader):
    if i > 0:
        break
    init_data = tdata


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 32))
        conv1_class = NNConv(d.num_features, 32, nn1, aggr='mean') \
            .jittable(x=init_data.x,
                      edge_index=init_data.edge_index,
                      edge_attr=init_data.edge_attr)
        self.conv1 = conv1_class(d.num_features, 32, nn1, aggr='mean')
        tempx = self.conv1(x=init_data.x,
                           edge_index=init_data.edge_index,
                           edge_attr=init_data.edge_attr)

        nn2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 2048))
        conv2_class = NNConv(32, 64, nn2, aggr='mean') \
            .jittable(x=tempx,
                      edge_index=init_data.edge_index,
                      edge_attr=init_data.edge_attr)
        self.conv2 = conv2_class(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, x, edge_index, edge_attr,
                pos, batch: Optional[torch.Tensor]):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        weight = normalized_cut_2d(edge_index, pos)
        cluster = graclus(edge_index, weight, x.size(0))

        x, edge_index, edge_attr, pos, batch = \
            max_pool_jit(cluster, x, edge_index, None, pos, batch)

        assert x is not None
        assert edge_index is not None
        assert edge_attr is not None

        x = F.elu(self.conv2(x, edge_index, edge_attr))
        weight = normalized_cut_2d(edge_index, pos)
        cluster = graclus(edge_index, weight, x.size(0))
        x, batch = max_pool_x(cluster, x, batch)

        assert batch is not None
        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.script(Net().to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data.x, data.edge_index, data.edge_attr,
                         data.pos, data.batch), data.y) \
            .backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr,
                     data.pos, data.batch) \
            .max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)


for epoch in range(1, 31):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
