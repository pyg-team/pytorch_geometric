import torch
from torch_geometric.utils import to_dense_batch


def test_to_dense_batch():
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    out, mask = to_dense_batch(x, batch)
    expected = [
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
        [[7, 8], [9, 10], [11, 12]],
    ]
    assert out.size() == (3, 3, 2)
    assert out.tolist() == expected
    assert mask.tolist() == [1, 1, 0, 1, 0, 0, 1, 1, 1]

    out = to_dense_batch(x)[0]
    assert out.size() == (1, 6, 2)
    assert out[0].tolist() == x.tolist()
