import pytest
import torch
import torch.nn as nn

from torch_geometric.llm.models import COFT, LLM, Granularity


class FakeLLM(LLM):
    """A minimal fake LLM that returns deterministic numeric scores.
    Loss = 0.1 * len(answer[0])
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = "fake-llm"

    def forward(self, question, answer):
        # deterministic fake score
        return torch.tensor(0.1 * len(answer[0]), dtype=torch.float32)


@pytest.fixture
def coft():
    trip = [("Q3", "R1", "Q1"), ("Q1", "R1", "Q2")]
    alias = {
        "Q1": ["apple", "red apple"],
        "Q2": ["banana"],
        "Q3": ["fruit"],
    }
    return COFT(FakeLLM(), trip, alias)


def test_recall(coft):
    out = coft._recall_candidates("I love eating fruit.")
    assert "fruit" in out
    assert "apple" in out
    assert "banana" not in out


def test_sentence_highlight(coft):
    text = ("apple is nice. banana is sweet. apple again. "
            "banana also good. apple delicious.")
    out = coft.highlight("fruit", text)
    assert "**" in out


def test_word_highlight(coft):
    out = coft.highlight("fruit", "banana and apple pineapple",
                         granularity=Granularity.WORD)
    print(out)
    assert "**apple**" in out or "**banana**" in out


def test_paragraph_highlight(coft):
    txt = "apple paragraph\nbanana paragraph"
    out = coft.highlight("fruit", txt, granularity=Granularity.PARAGRAPH)
    assert any(p.startswith("**") for p in out.split("\n"))
