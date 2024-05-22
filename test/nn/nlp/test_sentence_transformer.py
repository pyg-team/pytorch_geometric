import pytest

from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.testing import onlyFullTest, withCUDA


@withCUDA
@onlyFullTest
@pytest.mark.parametrize('batch_size', [None, 1])
def test_sentence_transformer(batch_size, device):
    model = SentenceTransformer(model_name='prajjwal1/bert-tiny').to(device)
    assert model.device == device
    assert str(model) == 'SentenceTransformer(model_name=prajjwal1/bert-tiny)'

    out = model.encode(
        [
            "this is a basic english text",
            "PyG is the best open-source GNN library :)",
        ],
        batch_size=batch_size,
    )
    assert out.is_cpu
    assert out.size() == (2, 128)

    out = model.encode(
        [
            "this is a basic english text",
            "PyG is the best open-source GNN library :)",
        ],
        batch_size=batch_size,
        output_device=device,
    )
    assert out.device == device
    assert out.size() == (2, 128)
