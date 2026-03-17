import gc

import pytest
import torch
from torch import Tensor

from torch_geometric.llm.models import LLM
from torch_geometric.llm.models.llm import get_llm_kwargs
from torch_geometric.testing import withPackage


def test_get_llm_kwargs():
    kwargs = get_llm_kwargs(required_memory=640)
    assert kwargs == {'revision': 'main'}


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


class DummyBatch(dict):
    """Mimics HuggingFace BatchEncoding."""
    def to(self, device):
        return self


class DummyTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors=None, padding=True):
        lengths = [len(t) for t in texts]
        max_len = max(lengths)

        ids = []
        mask = []

        for seq_len in lengths:
            padding = max_len - seq_len
            ids.append([0] * padding + list(range(1, seq_len + 1)))
            mask.append([0] * padding + [1] * seq_len)

        return DummyBatch({
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask)
        })


class DummyModel(torch.nn.Module):
    def get_input_embeddings(self):
        return torch.nn.Embedding(100, 8)

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        batch, seq, dim = inputs_embeds.shape

        class Out:
            pass

        out = Out()
        out.logits = torch.zeros(batch, seq, 10)
        return out


@pytest.fixture
def dummy_llm():
    llm = LLM.__new__(LLM)
    torch.nn.Module.__init__(llm)
    llm.device = torch.device("cpu")
    llm.tokenizer = DummyTokenizer()
    llm.model = DummyModel()
    return llm


@onlyRAG
def test_llm_prepare_inputs(dummy_llm):
    prompts = ["hello", "hi"]

    encoded = dummy_llm.tokenizer(prompts)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    emb = dummy_llm.model.get_input_embeddings()
    inputs_embeds = emb(input_ids)

    out = dummy_llm.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask)

    assert inputs_embeds.shape[0] == 2
    assert attention_mask.shape == input_ids.shape
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == inputs_embeds.shape[:2]


@onlyRAG
def test_llm_single_prompt(dummy_llm):
    encoded = dummy_llm.tokenizer(["test"])

    assert encoded["input_ids"].shape[0] == 1


@onlyRAG
def test_llm_variable_lengths(dummy_llm):
    prompts = ["a", "abcdef", "abc"]

    encoded = dummy_llm.tokenizer(prompts)

    input_ids = encoded["input_ids"]

    assert input_ids.shape[0] == 3
    assert input_ids.shape[1] == max(len(p) for p in prompts)
