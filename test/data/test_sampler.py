import sys
import random
import os.path as osp
import shutil

import torch
from torch.nn.functional import relu

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import SAGEConv
import torch_geometric.transforms as T


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
    class Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Net, self).__init__()
            self.conv1 = SAGEConv(in_channels, 16)
            self.conv2 = SAGEConv(16, 16)
            self.conv3 = SAGEConv(16, out_channels)

        def forward_data_flow(self, x, edge_weight, data_flow):
            block = data_flow[0]
            weight = None if block.e_id is None else edge_weight[block.e_id]
            x = relu(self.conv1(x, block.edge_index, weight, block.size))
            block = data_flow[1]
            weight = None if block.e_id is None else edge_weight[block.e_id]
            x = relu(self.conv2(x, block.edge_index, weight, block.size))
            block = data_flow[2]
            weight = None if block.e_id is None else edge_weight[block.e_id]
            x = self.conv3(x, block.edge_index, weight, block.size)
            return x

        def forward(self, x, edge_index, edge_weight):
            x = relu(self.conv1(x, edge_index, edge_weight))
            x = relu(self.conv2(x, edge_index, edge_weight))
            return self.conv3(x, edge_index, edge_weight)

    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = Planetoid(root, 'Cora')
    model = Net(dataset.num_features, dataset.num_classes)

    data1 = dataset[0]
    data1.edge_weight = None

    data2 = T.AddSelfLoops()(dataset[0])
    data2.edge_weight = torch.rand(data2.num_edges)

    data3 = dataset[0]
    loop = torch.stack([torch.arange(100, 200), torch.arange(100, 200)], dim=0)
    data3.edge_index = torch.cat([data3.edge_index, loop], dim=1)
    data3.edge_weight = None

    for data in [data1, data2, data3]:
        out_all = model(data.x, data.edge_index, data.edge_weight)

        loader = NeighborSampler(data, size=1.0, num_hops=3, batch_size=64,
                                 shuffle=False, drop_last=False,
                                 bipartite=True, add_self_loops=True)

        for data_flow in loader(data.train_mask):
            out = model.forward_data_flow(data.x[data_flow[0].n_id],
                                          data.edge_weight, data_flow)
            assert torch.allclose(out_all[data_flow.n_id], out)

        loader = NeighborSampler(data, size=1.0, num_hops=3, batch_size=64,
                                 shuffle=False, drop_last=False,
                                 bipartite=False)

        for subdata in loader(data.train_mask):
            weight = data.edge_weight
            weight = None if weight is None else weight[subdata.e_id]
            out = model(data.x[subdata.n_id], subdata.edge_index, weight)
            out = out[subdata.sub_b_id]
            assert torch.allclose(out_all[subdata.b_id], out)

    shutil.rmtree(root)
