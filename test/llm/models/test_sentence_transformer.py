import pytest
import torch

from torch_geometric.llm.models import (
    SentenceTransformer,
    last_pooling,
    mean_pooling,
)
from torch_geometric.testing import onlyRAG, withCUDA, withPackage


@withCUDA
@onlyRAG
@withPackage('transformers')
@pytest.mark.parametrize('batch_size', [None, 1])
@pytest.mark.parametrize('verbose', [True, False])
@pytest.mark.parametrize('pooling_strategy',
                         ['mean', 'last', 'cls', 'last_hidden_state'])
def test_sentence_transformer(batch_size, pooling_strategy, device, verbose):
    model = SentenceTransformer(
        model_name='prajjwal1/bert-tiny',
        pooling_strategy=pooling_strategy,
    ).to(device)
    assert model.device == device
    assert str(model) == 'SentenceTransformer(model_name=prajjwal1/bert-tiny)'

    text = [
        "this is a basic english text",
        "PyG is the best open-source GNN library :)",
    ]

    out = model.encode(text, batch_size=batch_size, output_device=device,
                       verbose=verbose)
    assert out.device == device
    if pooling_strategy == 'last_hidden_state':
        assert out.size() == (2, 7, 128)
    else:
        assert out.size() == (2, 128)

    out = model.encode([], batch_size=batch_size, output_device=device,
                       verbose=verbose)
    assert out.device == device
    if pooling_strategy == 'last_hidden_state':
        assert out.size() == (0, 3, 128)
    else:
        assert out.size() == (0, 128)

    input_ids, mask = model.get_input_ids(text, batch_size=batch_size,
                                          output_device=device)
    assert input_ids.size() == (2, 7)
    assert mask.size() == (2, 7)


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
