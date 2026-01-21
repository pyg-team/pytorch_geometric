import pytest
import torch

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider


class DummyBias(BaseBiasProvider):
    """Dummy implementation of BaseBiasProvider for testing purposes."""
    def _extract_raw_distances(self, data):
        """Return input data as distances (dummy implementation)."""
        return data

    def _embed_bias(self, distances: torch.LongTensor) -> torch.Tensor:
        """Return input distances as embeddings (dummy implementation)."""
        return distances


@pytest.fixture
def dummy_bias():
    """Fixture that returns a DummyBias instance for testing BaseBias
    assertions.
    """
    return DummyBias(num_heads=4, use_super_node=False)


def test_assert_square_invalid_dim(dummy_bias):
    """Test that _assert_square raises ValueError when tensor is not 3D."""
    with pytest.raises(ValueError):
        dummy_bias._assert_square(torch.zeros(2, 2))


def test_assert_square_non_square(dummy_bias):
    """Test that _assert_square raises ValueError when trailing dims are not
    square.
    """
    with pytest.raises(ValueError):
        dummy_bias._assert_square(torch.zeros(1, 3, 2))


def test_assert_embed_shape_invalid_dim(dummy_bias):
    """Test _assert_embed_shape raises ValueError when tensor is not 4D."""
    with pytest.raises(ValueError):
        dummy_bias._assert_embed_shape(torch.zeros(2, 3, 3))


def test_assert_embed_shape_wrong_heads(dummy_bias):
    """Test that _assert_embed_shape raises ValueError when last dimension
    does not equal num_heads.
    """
    with pytest.raises(ValueError):
        dummy_bias._assert_embed_shape(torch.zeros(2, 3, 3, 3))
