import pytest
import torch

from torch_geometric.nn.models.glem import deal_nan


def test_deal_nan_tensor_replaces_nans():
    x = torch.tensor([1.0, float('nan'), 3.0])
    result = deal_nan(x)

    expected = torch.tensor([1.0, 0.0, 3.0])
    assert torch.allclose(result, expected, equal_nan=True)
    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result).any()


def test_deal_nan_non_tensor_passthrough():
    assert deal_nan(42.0) == 42.0
    assert deal_nan("foo") == "foo"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deal_nan_tensor_dtypes(dtype):
    # Create a tensor with one NaN value
    x = torch.tensor([1.0, float('nan'), 3.0], dtype=dtype)
    result = deal_nan(x)

    expected = torch.tensor([1.0, 0.0, 3.0], dtype=dtype)

    # `bfloat16` doesn't support `allclose` directly on CPU,
    # so we cast to float32 for comparison
    if dtype == torch.bfloat16:
        assert torch.allclose(result.to(torch.float32),
                              expected.to(torch.float32), atol=1e-2)
    else:
        assert torch.allclose(result, expected, equal_nan=True)

    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result).any()
    assert result.dtype == dtype


def test_deal_nan_is_non_mutating():
    x = torch.tensor([1.0, float('nan'), 3.0])
    x_copy = x.clone()
    _ = deal_nan(x)
    assert torch.isnan(x).any()  # Original still contains NaN
    assert torch.allclose(x, x_copy, equal_nan=True)
