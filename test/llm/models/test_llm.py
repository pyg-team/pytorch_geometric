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

    model = LLM(
        model_name='HuggingFaceTB/SmolLM-360M',
        num_params=1,
        dtype=torch.float16,
    )
    assert str(model) == 'LLM(HuggingFaceTB/SmolLM-360M)'

    loss = model(question, answer)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert loss >= 0.0

    pred = model.inference(question)
    assert len(pred) == 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
