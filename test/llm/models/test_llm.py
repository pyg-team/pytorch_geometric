import gc

import pytest
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

        for l in lengths:
            pad = max_len - l
            ids.append([0]*pad + list(range(1, l+1)))
            mask.append([0]*pad + [1]*l)

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


def test_llm_prepare_inputs(dummy_llm):
    prompts = ["hello", "hi"]

    encoded = dummy_llm.tokenizer(prompts)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    emb = dummy_llm.model.get_input_embeddings()
    inputs_embeds = emb(input_ids)

    out = dummy_llm.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )

    assert inputs_embeds.shape[0] == 2
    assert attention_mask.shape == input_ids.shape
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == inputs_embeds.shape[:2]


def test_llm_single_prompt(dummy_llm):
    encoded = dummy_llm.tokenizer(["test"])

    assert encoded["input_ids"].shape[0] == 1


def test_llm_variable_lengths(dummy_llm):
    prompts = ["a", "abcdef", "abc"]

    encoded = dummy_llm.tokenizer(prompts)

    input_ids = encoded["input_ids"]

    assert input_ids.shape[0] == 3
    assert input_ids.shape[1] == max(len(p) for p in prompts)
