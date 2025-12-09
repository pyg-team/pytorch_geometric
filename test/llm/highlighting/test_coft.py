import torch
import pytest

from torch_geometric.llm.highlighting.coft import COFT
from torch_geometric.llm.highlighting.granularity import Granularity
from torch_geometric.llm.highlighting.fake_llm import FakeLLM


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
    text = (
        "apple is nice. banana is sweet. apple again. "
        "banana also good. apple delicious."
    )
    out = coft.highlight("fruit", text)
    assert "**" in out


def test_word_highlight(coft):
    out = coft.highlight(
        "fruit",
        "banana and apple",
        granularity=Granularity.WORD,
    )
    assert ("**apple**" in out) or ("**banana**" in out)


def test_paragraph_highlight(coft):
    txt = "apple paragraph\nbanana paragraph"
    out = coft.highlight("fruit", txt, granularity=Granularity.PARAGRAPH)
    assert any(p.startswith("**") for p in out.split("\n"))
