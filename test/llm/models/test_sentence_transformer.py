import pytest
import torch

from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.llm.models.sentence_transformer import (
    last_pooling,
    mean_pooling,
)
from torch_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('transformers')
@pytest.mark.parametrize('batch_size', [None, 1])
@pytest.mark.parametrize('pooling_strategy', ['mean', 'last', 'cls'])
@pytest.mark.parametrize('verbose', [True, False])
def test_sentence_transformer(batch_size, pooling_strategy, device, verbose):

    model_name = 'bert-base-uncased'
    model = SentenceTransformer(
        model_name=model_name,
        pooling_strategy=pooling_strategy,
    ).to(device)
    assert model.device == device
    assert str(model) == f'SentenceTransformer(model_name={model_name})'

    text = [
        "this is a basic english text",
        "PyG is the best open-source GNN library :)",
    ]

    model_embedding_dim = model.model.config.hidden_size

    out = model.encode(text, batch_size=batch_size, verbose=verbose)
    assert out.device == device
    assert out.shape == (2, model_embedding_dim)

    out = model.encode(text, batch_size=batch_size, output_device='cpu',
                       verbose=verbose)
    assert out.is_cpu
    assert out.shape == (2, model_embedding_dim)

    out = model.encode([], batch_size=batch_size, verbose=verbose)
    assert out.device == device
    assert out.shape == (0, model_embedding_dim)


def test_mean_pooling():
    x = torch.randn(2, 1, 2)
    attention_mask = torch.zeros(2, 1)

    result = mean_pooling(x, attention_mask)
    expected = torch.zeros_like(x)
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize('mask', [torch.ones, torch.zeros])
def test_last_pooling(mask):
    x = torch.randn(2, 1, 2)
    attention_mask = mask(2, 1, dtype=torch.long)
    out = last_pooling(x, attention_mask)
    assert torch.allclose(out, x[:, 0, :], atol=1e-6)
