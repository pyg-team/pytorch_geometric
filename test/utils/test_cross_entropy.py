import pytest
import torch
import torch.nn.functional as F

from torch_geometric.testing import withCUDA
from torch_geometric.utils.cross_entropy import sparse_cross_entropy


@withCUDA
@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multiclass(
    with_edge_label_weight: bool,
    device: torch.device,
) -> None:
    x = torch.randn(5, 5, device=device, requires_grad=True)
    y = torch.eye(5, device=device)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = torch.rand(edge_label_index.size(1), device=device)
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)


@withCUDA
@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multilabel(
    with_edge_label_weight: bool,
    device: torch.device,
) -> None:
    x = torch.randn(4, 4, device=device, requires_grad=True)
    y = torch.randint_like(x, 0, 2)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = torch.rand(edge_label_index.size(1), device=device)
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)


@withCUDA
@pytest.mark.parametrize('edge_label_weights', [
    [2.0, -10.0, 1.0, -5.0, 4.0, 0.0, -1.0],
    [-2.0, -1.0, -1.0, -3.0, -4.0, -10.0, -1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
def test_sparse_cross_entropy_negative_weight(
    edge_label_weights: list[float],
    device: torch.device,
) -> None:
    x = torch.randn(4, 8, device=device, requires_grad=True)
    edge_label_index = torch.tensor([
        [0, 0, 1, 2, 2, 2, 3],
        [2, 4, 6, 5, 3, 1, 1],
    ], device=device)
    edge_label_weight = torch.tensor(edge_label_weights, device=device)
    pos_mask = edge_label_weight >= 0

    y = torch.zeros_like(x)
    y[
        edge_label_index[0, pos_mask],
        edge_label_index[1, pos_mask],
    ] = edge_label_weight[pos_mask]

    _x = x.clone()
    _x[
        edge_label_index[0, ~pos_mask],
        edge_label_index[1, ~pos_mask],
    ] += edge_label_weight[~pos_mask].abs().log()
    expected = F.cross_entropy(_x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)
