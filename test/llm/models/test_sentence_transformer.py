import pytest

from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.testing import onlyRAG, withCUDA, withPackage


@withCUDA
@onlyRAG
@withPackage('transformers')
@pytest.mark.parametrize('batch_size', [None, 1])
@pytest.mark.parametrize('pooling_strategy', ['mean', 'last', 'cls'])
def test_sentence_transformer(batch_size, pooling_strategy, device):

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

    out = model.encode(text, batch_size=batch_size)
    assert out.device == device
    assert out.shape == (2, model_embedding_dim)

    out = model.encode(text, batch_size=batch_size, output_device='cpu')
    assert out.is_cpu
    assert out.shape == (2, model_embedding_dim)

    out = model.encode([], batch_size=batch_size)
    assert out.device == device
    assert out.shape == (0, model_embedding_dim)
