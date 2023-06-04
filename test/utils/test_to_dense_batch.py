from typing import Tuple

import pytest
import torch
from torch import Tensor

from torch_geometric.experimental import set_experimental_mode
from torch_geometric.testing import onlyFullTest
from torch_geometric.utils import to_dense_batch


@pytest.mark.parametrize('fill', [70.0, torch.tensor(49.0)])
def test_to_dense_batch(fill):
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    item = fill.item() if isinstance(fill, Tensor) else fill
    expected = torch.Tensor([
        [[1, 2], [3, 4], [item, item]],
        [[5, 6], [item, item], [item, item]],
        [[7, 8], [9, 10], [11, 12]],
    ])

    out, mask = to_dense_batch(x, batch, fill_value=fill)
    assert out.size() == (3, 3, 2)
    assert torch.equal(out, expected)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=2, fill_value=fill)
    assert out.size() == (3, 2, 2)
    assert torch.equal(out, expected[:, :2])
    assert mask.tolist() == [[1, 1], [1, 0], [1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=5, fill_value=fill)
    assert out.size() == (3, 5, 2)
    assert torch.equal(out[:, :3], expected)
    assert mask.tolist() == [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]

    out, mask = to_dense_batch(x, fill_value=fill)
    assert out.size() == (1, 6, 2)
    assert torch.equal(out[0], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=2, fill_value=fill)
    assert out.size() == (1, 2, 2)
    assert torch.equal(out[0], x[:2])
    assert mask.tolist() == [[1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=10, fill_value=fill)
    assert out.size() == (1, 10, 2)
    assert torch.equal(out[0, :6], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

    out, mask = to_dense_batch(x, batch, batch_size=4, fill_value=fill)
    assert out.size() == (4, 3, 2)


def test_to_dense_batch_disable_dynamic_shapes():
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    with set_experimental_mode(True, 'disable_dynamic_shapes'):
        with pytest.raises(ValueError, match="'batch_size' needs to be set"):
            out, mask = to_dense_batch(x, batch, max_num_nodes=6)
        with pytest.raises(ValueError, match="'max_num_nodes' needs to be"):
            out, mask = to_dense_batch(x, batch, batch_size=4)
        with pytest.raises(ValueError, match="'batch_size' needs to be set"):
            out, mask = to_dense_batch(x)

        out, mask = to_dense_batch(x, batch_size=1, max_num_nodes=6)
        assert out.size() == (1, 6, 2)
        assert mask.size() == (1, 6)

        out, mask = to_dense_batch(x, batch, batch_size=3, max_num_nodes=10)
        assert out.size() == (3, 10, 2)
        assert mask.size() == (3, 10)


@onlyFullTest
def test_to_dense_batch_jit():
    @torch.jit.script
    def to_dense_batch_jit(
        x: Tensor,
        batch: Tensor,
        fill_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return to_dense_batch(x, batch, fill_value=fill_value)

    x = torch.randn(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    out, mask = to_dense_batch_jit(x, batch, fill_value=torch.tensor(0.0))
    assert out.size() == (3, 3, 2)
    assert mask.size() == (3, 3)
