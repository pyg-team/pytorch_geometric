import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import to_dense_batch


def test_to_dense_batch():
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    expected = torch.Tensor([
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
        [[7, 8], [9, 10], [11, 12]],
    ])

    out, mask = to_dense_batch(x, batch)
    assert out.size() == (3, 3, 2)
    assert torch.equal(out, expected)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    if is_full_test():
        jit = torch.jit.script(to_dense_batch)
        out, mask = jit(x, batch)
        assert torch.equal(out, expected)
        assert mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=2)
    assert out.size() == (3, 2, 2)
    assert torch.equal(out, expected[:, :2])
    assert mask.tolist() == [[1, 1], [1, 0], [1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=5)
    assert out.size() == (3, 5, 2)
    assert torch.equal(out[:, :3], expected)
    assert mask.tolist() == [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]

    out, mask = to_dense_batch(x)
    assert out.size() == (1, 6, 2)
    assert torch.equal(out[0], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=2)
    assert out.size() == (1, 2, 2)
    assert torch.equal(out[0], x[:2])
    assert mask.tolist() == [[1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=10)
    assert out.size() == (1, 10, 2)
    assert torch.equal(out[0, :6], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

    out, mask = to_dense_batch(x, batch, batch_size=4)
    assert out.size() == (4, 3, 2)
