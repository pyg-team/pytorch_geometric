import torch
from torch_geometric.data import (Data, GraphSAINTNodeSampler,
                                  GraphSAINTEdgeSampler,
                                  GraphSAINTRandomWalkSampler)


def test_graph_saint():
    adj = torch.tensor([
        [1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])

    edge_index = adj.nonzero().t()
    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    data = Data(edge_index=edge_index, x=x, num_nodes=6)

    torch.manual_seed(12345)
    loader = GraphSAINTNodeSampler(data, batch_size=2, num_steps=4,
                                   sample_coverage=10, log=False)

    for sample in loader:
        assert len(sample) == 4
        assert sample.num_nodes <= 2
        assert sample.num_edges <= 3 * 2
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    torch.manual_seed(12345)
    loader = GraphSAINTEdgeSampler(data, batch_size=2, num_steps=4,
                                   sample_coverage=10, log=False)

    for sample in loader:
        assert len(sample) == 4
        assert sample.num_nodes <= 4
        assert sample.num_edges <= 3 * 4
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    torch.manual_seed(12345)
    loader = GraphSAINTRandomWalkSampler(data, batch_size=2, walk_length=1,
                                         num_steps=4, sample_coverage=10,
                                         log=False)

    for sample in loader:
        assert len(sample) == 4
        assert sample.num_nodes <= 4
        assert sample.num_edges <= 3 * 4
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges
