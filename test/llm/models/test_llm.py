import gc

import torch
from torch import Tensor

from torch_geometric.llm.models import LLM
from torch_geometric.testing import onlyRAG, withPackage


@onlyRAG
@withPackage('transformers', 'accelerate')
def test_llm() -> None:
    question = ['Is PyG the best open-source GNN library?']
    answer = ['yes!']

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


@onlyRAG
@withPackage('transformers', 'accelerate')
def test_llm_uncertainty() -> None:
    """Test LLM with uncertainty estimation."""
    question = ['What is the capital of France?']

    # Test with uncertainty enabled
    model = LLM(
        model_name='HuggingFaceTB/SmolLM-360M',
        num_params=1,
        dtype=torch.float16,
        uncertainty_estim=True,
        uncertainty_cfg={
            'h_star': 0.05,
            'isr_threshold': 1.0,
            'm': 3,
            'n_samples': 2,
            'B_clip': 12.0,
            'clip_mode': 'one-sided',
            'skeleton_policy': 'auto',
            'temperature': 0.5,
            'max_tokens_decision': 8,
            'backend': 'hf',
            'mask_refusals_in_loss': True,
        },
    )

    assert model.uncertainty_estim is True

    # Test inference with uncertainty
    result = model.inference(
        question=question,
        context=[''],
        max_tokens=10,
        return_uncertainty=True,
    )

    # Should return tuple of (texts, uncertainties)
    assert isinstance(result, tuple)
    texts, uncertainties = result
    assert len(texts) == 1
    assert len(uncertainties) == 1

    # Check uncertainty object has expected attributes
    uncertainty = uncertainties[0]
    assert hasattr(uncertainty, 'isr')
    assert hasattr(uncertainty, 'decision_answer')
    assert isinstance(uncertainty.isr, float)
    assert isinstance(uncertainty.decision_answer, bool)

    del model
    gc.collect()
    torch.cuda.empty_cache()
