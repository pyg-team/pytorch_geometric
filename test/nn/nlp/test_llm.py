import torch
from torch import Tensor

from torch_geometric.nn.nlp import LLM
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('transformers', 'accelerate')
def test_llm() -> None:
    question = ["Is PyG the best open-source GNN library?"]
    answer = ["yes!"]

    model = LLM(
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        num_params=1,
        dtype=torch.float16,
    )
    assert str(model) == 'LLM(TinyLlama/TinyLlama-1.1B-Chat-v0.1)'

    loss = model(question, answer)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert loss >= 0.0

    pred = model.inference(question)
    assert len(pred) == 1
