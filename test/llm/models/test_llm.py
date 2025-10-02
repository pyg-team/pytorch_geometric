import gc

import torch
from torch import Tensor

from torch_geometric.llm.models import LLM
from torch_geometric.testing import onlyRAG, withPackage


@onlyRAG
@withPackage('transformers', 'accelerate')
def test_llm() -> None:
    question = ["Is PyG the best open-source GNN library?"]
    answer = ["yes!"]

    model = LLM(model_name='Qwen/Qwen3-0.6B', num_params=1,
                dtype=torch.float16,
                sys_prompt="You're an agent, answer my questions.")
    assert str(model) == 'LLM(Qwen/Qwen3-0.6B)'

    loss = model(question, answer)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert loss >= 0.0

    pred = model.inference(question)
    assert len(pred) == 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
