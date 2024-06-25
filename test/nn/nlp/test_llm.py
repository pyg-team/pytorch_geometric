import pytest

from torch_geometric.testing import onlyFullTest, withCUDA, withPackage
from torch_geometric.nn.nlp.llm import LLM
from torch_geometric.data import Data

@withCUDA
@onlyFullTest
@withPackage('transformers')
def test_llm():
    batch = Data()
    batch.question = ["Is PyG the best open-source GNN library?"]
    batch.label = ["yes!"]

    model = LLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1", num_params=1)

    # test train
    loss = model(batch.question, batch.label)
    assert loss is not None

    # test inference
    out = model.inference(batch.question)
    assert out['pred'] is not None