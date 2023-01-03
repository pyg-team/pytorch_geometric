import pytest
import torch

from torch_geometric.utils import from_nested_tensor, to_nested_tensor


def test_to_nested_tensor():
    x = torch.randn(5, 4, 3)

    out = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    out = out.to_padded_tensor(padding=0)
    assert out.size() == (2, 3, 4, 3)
    assert torch.allclose(out[0, :2], x[0:2])
    assert torch.allclose(out[1, :3], x[2:5])

    out = to_nested_tensor(x, ptr=torch.tensor([0, 2, 5]))
    out = out.to_padded_tensor(padding=0)
    assert out.size() == (2, 3, 4, 3)
    assert torch.allclose(out[0, :2], x[0:2])
    assert torch.allclose(out[1, :3], x[2:5])

    out = to_nested_tensor(x)
    out = out.to_padded_tensor(padding=0)
    assert out.size() == (1, 5, 4, 3)
    assert torch.allclose(out[0], x)


def test_from_nested_tensor():
    x = torch.randn(5, 4, 3)

    nested = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    out, batch = from_nested_tensor(nested)

    assert torch.equal(x, out)
    assert batch.tolist() == [0, 0, 1, 1, 1]


def test_to_and_from_nested_tensor_autograd():
    x = torch.randn(5, 4, 3, requires_grad=True)

    out = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    with pytest.raises(NotImplementedError, match="requires gradients"):
        out, _ = from_nested_tensor(out)
        out.backward(torch.randn_like(out))
        assert x.grad is not None
