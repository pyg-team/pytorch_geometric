import torch

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data, NeighborSampler


def test_sampler():
    torch.manual_seed(12345)

    num_nodes = 10
    data = Data(edge_index=erdos_renyi_graph(num_nodes, 0.1))
    data.num_nodes = num_nodes

    loader = NeighborSampler(
        data, size=[4, 0.5], num_hops=2, batch_size=2, shuffle=True)

    for data_flow in loader():
        assert data_flow.__repr__() == 'DataFlow(2<-1<-1)'
        assert data_flow.n_id.tolist() == [0, 7]
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

    loader = NeighborSampler(
        data,
        size=[4, 0.5],
        num_hops=2,
        batch_size=3,
        drop_last=True,
        shuffle=False,
        add_self_loops=True)

    for data_flow in loader():
        pass
    for data_flow in loader(torch.tensor([0, 1, 2, 3, 4])):
        pass
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    for data_flow in loader(mask):
        pass
