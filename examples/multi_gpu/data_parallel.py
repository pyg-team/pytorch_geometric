import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import (
    DataParallel,
    NNConv,
    SplineConv,
    global_mean_pool,
)
from torch_geometric.typing import WITH_TORCH_SPLINE_CONV

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data', 'MNIST')
dataset = MNISTSuperpixels(path, transform=T.Cartesian()).shuffle()
loader = DataListLoader(dataset, batch_size=1024, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if WITH_TORCH_SPLINE_CONV:
            self.conv1 = SplineConv(dataset.num_features, 32, dim=2,
                                    kernel_size=5)
            self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        else:
            nn1 = Sequential(Linear(2, 25), ReLU(),
                             Linear(25, dataset.num_features * 32))
            self.conv1 = NNConv(dataset.num_features, 32, nn1, aggr='mean')

            nn2 = Sequential(Linear(2, 25), ReLU(), Linear(25, 32 * 64))
            self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, dataset.num_classes)

    def forward(self, data):
        print(f'Inside model - num graphs: {data.num_graphs}, '
              f'device: {data.batch.device}')

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)


model = Net()
print(f"Let's use {torch.cuda.device_count()} GPUs!")
model = DataParallel(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for data_list in loader:
    optimizer.zero_grad()
    output = model(data_list)
    print(f'Outside model - num graphs: {output.size(0)}')
    y = torch.cat([data.y for data in data_list]).to(output.device)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
