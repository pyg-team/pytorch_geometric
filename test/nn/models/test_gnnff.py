import torch

from torch_geometric.nn import GNNFF
from torch_geometric.testing import is_full_test, withPackage


@withPackage('torch_sparse')  # TODO `triplet` requires `SparseTensor` for now.
def test_gnnff():
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)

    model = GNNFF(
        hidden_node_channels=5,
        hidden_edge_channels=5,
        num_layers=5,
    )
    model.reset_parameters()

    out = model(z, pos)
    assert out.size() == (20, 3)

    if is_full_test():
        jit = torch.jit.export(model)
        assert torch.allclose(jit(z, pos), out)
