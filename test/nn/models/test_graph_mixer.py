import torch

from torch_geometric.nn.models.graph_mixer import (
    LinkEncoder,
    NodeEncoder,
    get_latest_k_edge_attrs,
)


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


def test_link_encoding():
    num_nodes = 3
    num_edges = 6
    num_edge_features = 10
    edge_attr = torch.rand((num_edges, num_edge_features))
    edge_index = torch.randint(low=0, high=num_nodes, size=(2, num_edges))
    edge_time = torch.rand(num_edges)

    K = 3
    hidden_channels = 7
    out_channels = 11
    time_channels = 13
    dropout = 0.5

    encoder = LinkEncoder(
        K=K,
        in_channels=num_edge_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        time_channels=time_channels,
        dropout=dropout,
    )
    assert str(encoder) == (f'LinkEncoder(K={K}, '
                            f'in_channels={num_edge_features}, '
                            f'hidden_channels={hidden_channels}, '
                            f'out_channels={out_channels}, '
                            f'time_channels={time_channels}, '
                            f'dropout={dropout})')

    out = encoder(edge_attr, edge_time, edge_index)
    assert out.size() == (num_nodes, out_channels)


def test_latest_k_edge_attr():
    num_nodes = 3
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 0], [0, 1, 0, 1, 0, 1, 2]])
    edge_time = torch.tensor([3, 1, 2, 3, 1, 2, 3])
    edge_attr = torch.tensor([1, -1, 3, 4, -1, 6, 7]).view(-1, 1)

    k = 2
    latest_k_edge_attrs = get_latest_k_edge_attrs(k, edge_index, edge_time,
                                                  edge_attr, num_nodes)
    expected_output = torch.tensor([[[1], [3]], [[4], [6]], [[7], [0]]])
    assert latest_k_edge_attrs.shape == (3, 2, 1)
    assert latest_k_edge_attrs.equal(expected_output)

    k = 1
    latest_k_edge_attrs = get_latest_k_edge_attrs(k, edge_index, edge_time,
                                                  edge_attr, num_nodes)
    expected_output = torch.tensor([[[1]], [[4]], [[7]]])
    assert latest_k_edge_attrs.shape == (3, 1, 1)
    assert latest_k_edge_attrs.equal(expected_output)
