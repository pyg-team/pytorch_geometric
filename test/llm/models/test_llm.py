import gc

import pytest
import torch
from torch import Tensor

from torch_geometric.llm.models import LLM
from torch_geometric.llm.models.llm import get_llm_kwargs
from torch_geometric.testing import onlyRAG, withPackage


def test_get_llm_kwargs():
    kwargs = get_llm_kwargs(required_memory=640)
    assert kwargs == {'revision': 'main'}


@onlyRAG
@withPackage('transformers', 'accelerate')
@pytest.mark.parametrize('sys_prompt',
                         ['You are an agent, answer my questions.', None])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('context', [['This is context.'], None])
@pytest.mark.parametrize('use_embedding', [True, False])
def test_llm(sys_prompt, dtype, context, use_embedding) -> None:
    question = ["Is PyG the best open-source GNN library?"]
    answer = ["yes!"]

    model = LLM(
        model_name='Qwen/Qwen3-0.6B',
        num_params=1,
        dtype=dtype,
        sys_prompt=sys_prompt,
    )
    assert str(model) == 'LLM(Qwen/Qwen3-0.6B)'

    embedding = [torch.randn(1, 1024).to(model.device)
                 ] if use_embedding else None
    loss = model(question, answer, context=context, embedding=embedding)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert loss >= 0.0

    pred = model.inference(question)
    assert len(pred) == 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
