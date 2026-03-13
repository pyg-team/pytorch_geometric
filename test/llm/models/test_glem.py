import types

import pytest
import torch

from torch_geometric.llm.models.glem import GLEM, deal_nan
from torch_geometric.testing import withPackage


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


class DummyGNN(torch.nn.Module):
    out_channels = 3

    def forward(self, x, edge_index):
        return torch.zeros((1, self.out_channels))


class DummyLM(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = types.SimpleNamespace(
            pad_token_id=None,
            eos_token_id=0,
        )
        self.device = torch.device("cpu")

    def forward(self, **kwargs):
        return types.SimpleNamespace(logits=torch.zeros((1, self.num_labels)))


@withPackage('transformers')
def test_glem_bert_tiny_branch(monkeypatch):
    """Covers the BertForSequenceClassification branch."""
    from transformers import BertForSequenceClassification

    def fake_from_pretrained(*args, **kwargs):
        return DummyLM(kwargs["num_labels"])

    monkeypatch.setattr(
        BertForSequenceClassification,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    model = GLEM(
        lm_to_use="prajjwal1/bert-tiny",
        gnn_to_use=DummyGNN(),
        out_channels=3,
        lm_use_lora=False,
    )

    assert isinstance(model.lm, DummyLM)
    assert model.lm.num_labels == 3


@withPackage('transformers')
def test_glem_auto_model_branch(monkeypatch):
    """Covers the AutoModelForSequenceClassification branch."""
    from transformers import AutoModelForSequenceClassification

    def fake_from_pretrained(*args, **kwargs):
        return DummyLM(kwargs["num_labels"])

    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    model = GLEM(
        lm_to_use="bert-base-uncased",
        gnn_to_use=DummyGNN(),
        out_channels=3,
        lm_use_lora=False,
    )

    assert isinstance(model.lm, DummyLM)
    assert model.lm.num_labels == 3
