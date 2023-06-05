import torch

from torch_geometric.nn.models.graph_mixer import NodeEncoder


def test_node_encoder():
    x = torch.arange(4, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor([[1, 2, 0, 0, 1, 3], [0, 0, 1, 2, 2, 2]])
    edge_time = torch.tensor([0, 1, 1, 1, 2, 3])
    seed_time = torch.tensor([2, 2, 2, 2])

    encoder = NodeEncoder(time_window=2)
    assert str(encoder) == 'NodeEncoder(time_window=2)'

    out = encoder(x, edge_index, edge_time, seed_time)
    # Node 0 aggregates information from node 2 (excluding node 1).
    # Node 1 aggregates information from node 0.
    # Node 2 aggregates information from node 0 and node 1 (exluding node 3).
    # Node 3 aggregates no information.
    expected = torch.tensor([
        [0 + 2],
        [1 + 0],
        [2 + 0.5 * (0 + 1)],
        [3],
    ])
    assert torch.allclose(out, expected)
