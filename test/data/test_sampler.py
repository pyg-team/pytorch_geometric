import sys
import random
import os.path as osp
import shutil

import torch
import torch.nn.functional as F

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import SAGEConv


def test_sampler():
    num_nodes = 10
    data = Data(edge_index=erdos_renyi_graph(num_nodes, 0.1))
    data.num_nodes = num_nodes

    loader = NeighborSampler(data, size=[4, 0.5], num_hops=2, batch_size=2,
                             shuffle=True)

    for data_flow in loader():
        assert data_flow.__repr__()[:8] == 'DataFlow'
        assert data_flow.n_id.size() == (2, )
        assert data_flow.batch_size == 2
        assert len(data_flow) == 2
        block = data_flow[0]
        assert block.__repr__()[:5] == 'Block'
        for block in data_flow:
            pass
        data_flow = data_flow.to(torch.long)
        break
    for data_flow in loader(torch.tensor([0, 1, 2, 3, 4])):
        pass

    loader = NeighborSampler(data, size=[4, 0.5], num_hops=2, batch_size=3,
                             drop_last=True, shuffle=False,
                             add_self_loops=True)

    for data_flow in loader():
        pass
    for data_flow in loader(torch.tensor([0, 1, 2, 3, 4])):
        pass
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    for data_flow in loader(mask):
        pass


def test_cora():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    # dataset = Planetoid('/tmp/Cora', 'Cora')
    dataset = Planetoid(root, 'Cora')
    data = dataset[0]

    loader = NeighborSampler(data, size=1.0, num_hops=2, batch_size=64,
                             shuffle=True, drop_last=False, bipartite=False)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = SAGEConv(dataset.num_features, 16)
            self.conv2 = SAGEConv(16, dataset.num_classes)

        def forward_all(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

        def forward_batch(self, x, data_flow):
            x = x[data_flow[0].n_id]
            x = F.relu(self.conv1(x, data_flow[0].edge_index))
            return self.conv2(x, data_flow[1].edge_index)

    model = Net()

    model.forward_all(data.x, data.edge_index)

    for data_flow in loader(torch.tensor([0])):
        model.forward_batch(data.x, data_flow)[data_flow.batched_n_id]
        break

    shutil.rmtree(root)
