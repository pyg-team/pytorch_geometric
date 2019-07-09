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
    dataset = Planetoid(root, 'Cora')
    data = dataset[0]

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = SAGEConv(dataset.num_features, 16)
            self.conv2 = SAGEConv(16, 16)
            self.conv3 = SAGEConv(16, dataset.num_classes)

        def forward_data_flow(self, x, data_flow):
            block = data_flow[0]
            x = F.relu(self.conv1(x, block.edge_index, size=block.size))
            block = data_flow[1]
            x = F.relu(self.conv2(x, block.edge_index, size=block.size))
            block = data_flow[2]
            x = self.conv3(x, block.edge_index, size=block.size)
            return x

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return self.conv3(x, edge_index)

    model = Net()

    out_all = model(data.x, data.edge_index)

    loader = NeighborSampler(data, size=1.0, num_hops=3, batch_size=64,
                             shuffle=False, drop_last=False, bipartite=True,
                             add_self_loops=True)

    for data_flow in loader(data.train_mask):
        out = model.forward_data_flow(data.x[data_flow[0].n_id], data_flow)
        assert torch.allclose(out_all[data_flow.n_id], out)

    loader = NeighborSampler(data, size=1.0, num_hops=3, batch_size=64,
                             shuffle=False, drop_last=False, bipartite=False)

    for subdata in loader(data.train_mask):
        out = model(data.x[subdata.n_id], subdata.edge_index)[subdata.sub_b_id]
        assert torch.allclose(out_all[subdata.b_id], out)

    shutil.rmtree(root)
