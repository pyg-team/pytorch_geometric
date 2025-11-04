import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from torch_geometric.llm.models.txt2kg import (
    TXT2KG,
    _chunk_text,
    _llm_then_python_parse,
    _multiproc_helper,
    _parse_n_check_triples,
)
from torch_geometric.testing import onlyRAG

# ────────────────────────
# Fixtures
# ────────────────────────


@pytest.fixture
def mock_cloud_llm(monkeypatch):
    """Mock cloud LLM call to return deterministic triple string."""
    def mock_fn(txt, **kwargs):
        return "('Paris', 'capital of', 'France')"

    monkeypatch.setattr(
        "torch_geometric.llm.models.txt2kg._chunk_to_triples_str_cloud",
        mock_fn)
    return mock_fn


@pytest.fixture
def mock_local_llm(monkeypatch):
    """Mock local LLM inference without loading real model."""
    mock_model = MagicMock()
    mock_model.inference.return_value = ["('Earth', 'orbits', 'Sun')"]
    mock_llm_class = MagicMock()
    mock_llm_class.return_value = mock_model
    monkeypatch.setattr("torch_geometric.llm.models.LLM", mock_llm_class)
    return mock_model


# ────────────────────────
# Tests for helper functions
# ────────────────────────


def test_chunk_text_empty():
    assert _chunk_text("", 10) == []


def test_chunk_text_short():
    assert _chunk_text("Hello world.", 50) == ["Hello world."]


def test_chunk_text_long():
    text = "This is sentence one. This is sentence two! And this is three?"
    chunks = _chunk_text(text, chunk_size=30)
    assert len(chunks) >= 2
    assert "sentence one." in chunks[0]
    # Should not break mid-word at end
    for chunk in chunks:
        assert len(chunk) <= 60  # generous upper bound


def test_chunk_text_no_punctuation():
    text = "word " * 200  # no .!?
    chunks = _chunk_text(text, chunk_size=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.strip()) > 0


@pytest.mark.parametrize("input_str,expected", [
    ("('A', 'B', 'C')\n('D', 'E', 'F')", [('A', 'B', 'C'), ('D', 'E', 'F')]),
    ("('X', 'rel', 'Y') ('Z', 'has', 'W')", [("'X'", "'rel'", "'Y'"),
                                             ("'Z'", "'has'", "'W'")]),
    ("note: ignore this", []),
    ("('Alice', '', 'Bob')", [("'Alice'", "''", "'Bob'")]),
    ("('', 'r', 'e')", [("''", "'r'", "'e'")]),
    ("('valid', 'relation', 'entity')\nnote: extra", [
        ('valid', 'relation', 'entity')
    ]),
])
def test_parse_n_check_triples(input_str, expected):
    result = _parse_n_check_triples(input_str)
    assert result == expected


def test_llm_then_python_parse():
    def fake_llm(txt, **kwargs):
        return "('Mars', 'has', 'moons')"

    chunks = ["dummy"]
    result = _llm_then_python_parse(chunks, py_fn=_parse_n_check_triples,
                                    llm_fn=fake_llm)
    assert result == [("'Mars'", "'has'", "'moons'")]


# ────────────────────────
# Tests for TXT2KG class
# ────────────────────────


@onlyRAG
def test_txt2kg_cloud_mode(mock_cloud_llm):
    kg = TXT2KG(NVIDIA_API_KEY="fake-key", local_LM=False)
    kg.add_doc_2_KG("Paris is the capital of France.")

    assert 0 in kg.relevant_triples
    triples = kg.relevant_triples[0]
    assert triples == []


def test_txt2kg_local_mode(mock_local_llm):
    kg = TXT2KG(local_LM=True)
    kg.add_doc_2_KG("The Earth orbits the Sun.")

    assert 0 in kg.relevant_triples
    triples = kg.relevant_triples[0]
    assert triples == []


@onlyRAG
def test_txt2kg_with_qa_pair(mock_cloud_llm):
    kg = TXT2KG(NVIDIA_API_KEY="fake", local_LM=False)
    qa = ("What is Paris?", "Capital of France")
    kg.add_doc_2_KG("Paris is the capital of France.", QA_pair=qa)

    assert qa in kg.relevant_triples
    assert kg.relevant_triples[qa] == []


@onlyRAG
def test_txt2kg_duplicate_qa_warning(capfd, mock_cloud_llm):
    kg = TXT2KG(NVIDIA_API_KEY="fake", local_LM=False)
    qa = ("Q", "A")
    kg.add_doc_2_KG("text1", QA_pair=qa)
    kg.add_doc_2_KG("text2", QA_pair=qa)

    captured = capfd.readouterr()
    assert "Warning: QA_Pair was already added" in captured.out


@onlyRAG
def test_txt2kg_save_kg(mock_cloud_llm):
    kg = TXT2KG(NVIDIA_API_KEY="fake", local_LM=False)
    kg.add_doc_2_KG("Test document.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        kg.save_kg(str(tmp_path))
        loaded = torch.load(tmp_path)
        assert loaded == {0: []}
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


# ────────────────────────
# Multiprocessing helper test
# ────────────────────────


def test_multiproc_helper_saves_output(monkeypatch):
    def mock_llm_fn(txt, **kwargs):
        return "('Proc', 'runs on', 'GPU')"

    monkeypatch.setattr(
        "torch_geometric.llm.models.txt2kg._chunk_to_triples_str_cloud",
        mock_llm_fn)

    in_chunks = {0: ["sample text"]}
    rank = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock torch.save to capture path
        saved_paths = []
        original_save = torch.save

        def mock_save(obj, f):
            saved_paths.append(f)
            # Save to our temp dir
            actual_path = os.path.join(tmpdir, os.path.basename(f))
            original_save(obj, actual_path)

        monkeypatch.setattr("torch_geometric.llm.models.txt2kg.torch.save",
                            mock_save)

        _multiproc_helper(rank=rank, in_chunks_per_proc=in_chunks,
                          py_fn=_parse_n_check_triples, llm_fn=mock_llm_fn,
                          NIM_KEY="fake-key", NIM_MODEL="fake-model",
                          ENDPOINT_URL="http://fake.url")

        assert len(saved_paths) == 1
        assert "outs_for_proc_0" in saved_paths[0]

        result = torch.load(os.path.join(tmpdir, "outs_for_proc_0"))
        assert result == [("'Proc'", "'runs on'", "'GPU'")]
