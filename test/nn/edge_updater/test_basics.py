import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.edge_updater import (
    Compose, GCNNorm, AddRemainingSelfLoops
)


def test_gcn_norm():
    torch.manual_seed(2)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))

    updater = GCNNorm()
    assert str(updater) == 'GCNNorm()'
    out1 = updater(edge_index[1], edge_index[0], value,
                   dim_size=4)['edge_weight']
    out2 = gcn_norm(edge_index, value, 4,
                    improved=False, add_self_loops=False)[1]
    assert out1.size() == out2.size()
    assert torch.allclose(out1, out2, atol=1e-6)

    updater = Compose((
        AddRemainingSelfLoops(),
        GCNNorm()
    ))
    assert str(updater) == 'Compose([' \
                           'AddRemainingSelfLoops(), ' \
                           'GCNNorm()' \
                           '])'
    out1 = updater(
        edge_index_i=edge_index[1],
        edge_index_j=edge_index[0],
        edge_weight=value,
        edge_attr=None, edge_type=None,
        num_nodes=4, dim_size=4
    )['edge_weight']
    out2 = gcn_norm(edge_index, value, 4,
                    improved=False, add_self_loops=True)[1]
    assert out1.size() == out2.size()
    assert torch.allclose(out1, out2, atol=1e-6)

    updater = Compose((
        AddRemainingSelfLoops(2),
        GCNNorm()
    ))
    assert str(updater) == 'Compose([' \
                           'AddRemainingSelfLoops(), ' \
                           'GCNNorm()' \
                           '])'
    out1 = updater(
        edge_index_i=edge_index[1],
        edge_index_j=edge_index[0],
        edge_weight=value,
        edge_attr=None, edge_type=None,
        num_nodes=4, dim_size=4
    )['edge_weight']
    out2 = gcn_norm(edge_index, value, 4,
                    improved=True, add_self_loops=True)[1]
    assert out1.size() == out2.size()
    assert torch.allclose(out1, out2, atol=1e-6)


# TODO test fill
