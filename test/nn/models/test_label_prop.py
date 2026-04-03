import torch

import torch_geometric.typing
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.typing import SparseTensor


def test_label_prop():
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = LabelPropagation(num_layers=2, alpha=0.5)
    assert str(model) == 'LabelPropagation(num_layers=2, alpha=0.5)'

    # Test without mask:
    out = model(y, edge_index)
    assert out.size() == (6, 3)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert torch.allclose(model(y, adj.t()), out)

    # Test with mask:
    out = model(y, edge_index, mask)
    assert out.size() == (6, 3)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(model(y, adj.t(), mask), out)

    # Test post step:
    out = model(y, edge_index, mask, post_step=lambda y: torch.zeros_like(y))
    assert torch.sum(out) == 0.

    # Test that labeled node values are preserved after propagation:
    y2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])
    edge_index2 = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    mask2 = torch.tensor(
        [False, True, True, True, True, True, True, True, True])
    edge_weight2 = torch.ones(9)
    model2 = LabelPropagation(num_layers=2, alpha=0.9)
    out2 = model2(y=y2, edge_index=edge_index2, mask=mask2,
                  edge_weight=edge_weight2)
    # Labeled nodes must retain their original one-hot values
    from torch_geometric.utils import one_hot
    y2_oh = one_hot(y2)
    assert torch.allclose(out2[mask2], y2_oh[mask2].float())
