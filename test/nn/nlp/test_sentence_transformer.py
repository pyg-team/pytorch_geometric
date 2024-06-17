import pytest

from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.testing import onlyFullTest, withCUDA, withPackage


@withCUDA
@onlyFullTest
@withPackage('transformers')
@pytest.mark.parametrize('batch_size', [None, 1])
@pytest.mark.parametrize('pooling_strategy', ['mean', 'last', 'cls'])
def test_sentence_transformer(batch_size, pooling_strategy, device):
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

    out = model.encode(text, batch_size=batch_size)
    assert out.is_cpu
    assert out.size() == (2, 128)

    out = model.encode(text, batch_size=batch_size, output_device=device)
    assert out.device == device
    assert out.size() == (2, 128)
