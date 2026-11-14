import pytest
import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import (
    contains_isolated_nodes,
    remove_isolated_nodes,
)


def test_contains_isolated_nodes():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    assert not contains_isolated_nodes(edge_index)
    assert contains_isolated_nodes(edge_index, num_nodes=3)

    if is_full_test():
        jit = torch.jit.script(contains_isolated_nodes)
        assert not jit(edge_index)
        assert jit(edge_index, num_nodes=3)

    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 0]])
    assert contains_isolated_nodes(edge_index)


def test_remove_isolated_nodes():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])

    out, _, mask = remove_isolated_nodes(edge_index)
    assert out.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert mask.tolist() == [1, 1]

    if is_full_test():
        jit = torch.jit.script(remove_isolated_nodes)
        out, _, mask = jit(edge_index)
        assert out.tolist() == [[0, 1, 0], [1, 0, 0]]
        assert mask.tolist() == [1, 1]

    out, _, mask = remove_isolated_nodes(edge_index, num_nodes=3)
    assert out.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert mask.tolist() == [1, 1, 0]

    edge_index = torch.tensor([[0, 2, 1, 0, 2], [2, 0, 1, 0, 2]])
    edge_attr = torch.tensor([1, 2, 3, 4, 5])
    out1, out2, mask = remove_isolated_nodes(edge_index, edge_attr)
    assert out1.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
    assert out2.tolist() == [1, 2, 4, 5]
    assert mask.tolist() == [1, 0, 1]


@pytest.mark.parametrize('feature_dim', [None, 1, 2])
def test_remove_isolated_nodes_duplicate_self_loops(feature_dim):
    edge_index = torch.tensor([[3, 1, 0, 3, 1, 3, 1], [1, 3, 0, 3, 1, 3, 1]])
    edge_attr = None
    if feature_dim is not None:
        edge_attr = torch.arange(7 * feature_dim, dtype=torch.float)
        if feature_dim > 1:
            edge_attr = edge_attr.view(7, feature_dim)
        edge_attr.requires_grad_()

    out, attr, mask = remove_isolated_nodes(edge_index, edge_attr, num_nodes=5)
    assert mask.tolist() == [False, True, False, True, False]
    assert out.tolist() == [[1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1]]
    if edge_attr is None:
        assert attr is None
    else:
        expected = edge_attr[[0, 1, 4, 6, 3, 5]]
        assert torch.equal(attr, expected)
        attr.sum().backward()
        expected_grad = torch.ones_like(edge_attr)
        expected_grad[2] = 0
        assert torch.equal(edge_attr.grad, expected_grad)

    if is_full_test():
        jit = torch.jit.script(remove_isolated_nodes)
        jit_out, jit_attr, jit_mask = jit(edge_index, edge_attr, num_nodes=5)
        assert torch.equal(jit_out, out)
        assert torch.equal(jit_mask, mask)
        if edge_attr is not None:
            assert torch.equal(jit_attr, attr)
