import pytest
import torch

from torch_geometric.testing import is_full_test, withCUDA
from torch_geometric.utils import normalized_cut


@pytest.mark.parametrize('feature_shape', [(2, ), (4, ), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@withCUDA
def test_normalized_cut_edge_features(feature_shape, dtype, device):
    edge_index = torch.tensor([[1, 1, 2, 3], [3, 3, 1, 2]], device=device)
    num_features = int(torch.tensor(feature_shape).prod())
    edge_attr = torch.arange(1, 4 * num_features + 1, dtype=dtype,
                             device=device)
    edge_attr = edge_attr.view(4, *feature_shape).requires_grad_()
    # Endpoint degrees give one scale per edge, regardless of feature shape.
    scale = torch.tensor([1.5, 1.5, 2.0, 1.5], dtype=dtype, device=device)
    scale = scale.view([4] + [1] * len(feature_shape))
    expected = edge_attr * scale

    out = normalized_cut(edge_index, edge_attr)
    torch.testing.assert_close(out, expected)
    assert out.dtype == dtype
    assert out.shape == edge_attr.shape
    out.sum().backward()
    torch.testing.assert_close(edge_attr.grad, scale.expand_as(edge_attr))

    if is_full_test():
        jit = torch.jit.script(normalized_cut)
        torch.testing.assert_close(jit(edge_index, edge_attr), expected)


@pytest.mark.parametrize('feature_shape', [(2, ), (2, 3)])
def test_normalized_cut_empty_edge_features(feature_shape):
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, *feature_shape))
    out = normalized_cut(edge_index, edge_attr, num_nodes=4)
    assert out.shape == edge_attr.shape
    assert out.dtype == edge_attr.dtype


def test_normalized_cut():
    row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
    edge_attr = torch.tensor(
        [3.0, 3.0, 6.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 2.0])
    expected = torch.tensor([4.0, 4.0, 5.0, 2.5, 5.0, 1.0, 2.5, 2.0, 1.0, 2.0])

    out = normalized_cut(torch.stack([row, col], dim=0), edge_attr)
    assert torch.allclose(out, expected)

    if is_full_test():
        jit = torch.jit.script(normalized_cut)
        out = jit(torch.stack([row, col], dim=0), edge_attr)
        assert torch.allclose(out, expected)
