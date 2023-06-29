import pytest
import torch

from torch_geometric.testing import withPackage
from torch_geometric.utils import from_nested_tensor, to_nested_tensor


@withPackage('torch>=1.13.0')
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


@withPackage('torch>=1.13.0')
def test_from_nested_tensor():
    x = torch.randn(5, 4, 3)

    nested = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    out, batch = from_nested_tensor(nested, return_batch=True)

    assert torch.equal(x, out)
    assert batch.tolist() == [0, 0, 1, 1, 1]

    nested = torch.nested.nested_tensor([torch.randn(4, 3), torch.randn(5, 4)])
    with pytest.raises(ValueError, match="have the same size in dimension 1"):
        from_nested_tensor(nested)

    # Test zero-copy:
    nested = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    out = from_nested_tensor(nested)
    out += 1  # Increment in-place (which should increment `nested` as well).
    assert torch.equal(nested.to_padded_tensor(padding=0)[0, :2], out[0:2])
    assert torch.equal(nested.to_padded_tensor(padding=0)[1, :3], out[2:5])


@withPackage('torch>=1.13.0')
def test_to_and_from_nested_tensor_autograd():
    x = torch.randn(5, 4, 3, requires_grad=True)
    grad = torch.randn_like(x)

    out = to_nested_tensor(x, batch=torch.tensor([0, 0, 1, 1, 1]))
    out = from_nested_tensor(out)
    out.backward(grad)
    assert torch.equal(x.grad, grad)
