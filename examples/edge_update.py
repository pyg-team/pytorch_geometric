import os.path as osp

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]

class EdgeMessagePassingExample(MessagePassing):

    def forward(self, data: Data):
        x = self.propagate(data.edge_index, edge_attr=data.edge_attr, x=data.x)
        edge_attr = self.edge_update(data.edge_index, edge_attr=data.edge_attr, x=data.x)
        return x, edge_attr

    def edge_message(self, edge_attr, x_i, x_j):
        return x_i

    def message(self, edge_attr, x_i, x_j):
        return x_i

model = EdgeMessagePassingExample()
print(model.forward(data))
