import torch
from torch_geometric.data import (Data, GraphSAINTNodeSampler,
                                  GraphSAINTEdgeSampler,
                                  GraphSAINTRandomWalkSampler)


def test_graph_saint():
    adj = torch.tensor([
        [+1, +2, +3, +0, +4, +0],
        [+5, +6, +0, +7, +0, +8],
        [+9, +0, 10, +0, 11, +0],
        [+0, 12, +0, 13, +0, 14],
        [15, +0, 16, +0, 17, +0],
        [+0, 18, +0, 19, +0, 20],
    ])

    edge_index = adj.nonzero(as_tuple=False).t()
    edge_type = adj[edge_index[0], edge_index[1]]
    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    data = Data(edge_index=edge_index, x=x, edge_type=edge_type, num_nodes=6)

    torch.manual_seed(12345)
    loader = GraphSAINTNodeSampler(data, batch_size=3, num_steps=4,
                                   sample_coverage=10, log=False)

    sample = next(iter(loader))
    assert sample.x.tolist() == [[2, 2], [4, 4], [5, 5]]
    assert sample.edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]
    assert sample.edge_type.tolist() == [10, 11, 16, 17, 20]

    assert len(loader) == 4
    for sample in loader:
        assert len(sample) == 5
        assert sample.num_nodes <= 3
        assert sample.num_edges <= 3 * 4
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    torch.manual_seed(12345)
    loader = GraphSAINTEdgeSampler(data, batch_size=2, num_steps=4,
                                   sample_coverage=10, log=False)

    sample = next(iter(loader))
    assert sample.x.tolist() == [[0, 0], [2, 2], [3, 3]]
    assert sample.edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]
    assert sample.edge_type.tolist() == [1, 3, 9, 10, 13]

    assert len(loader) == 4
    for sample in loader:
        assert len(sample) == 5
        assert sample.num_nodes <= 4
        assert sample.num_edges <= 4 * 4
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    torch.manual_seed(12345)
    loader = GraphSAINTRandomWalkSampler(data, batch_size=2, walk_length=1,
                                         num_steps=4, sample_coverage=10,
                                         log=False)

    sample = next(iter(loader))
    assert sample.x.tolist() == [[1, 1], [2, 2], [4, 4]]
    assert sample.edge_index.tolist() == [[0, 1, 1, 2, 2], [0, 1, 2, 1, 2]]
    assert sample.edge_type.tolist() == [6, 10, 11, 16, 17]

    assert len(loader) == 4
    for sample in loader:
        assert len(sample) == 5
        assert sample.num_nodes <= 4
        assert sample.num_edges <= 4 * 4
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges
