import sys
import types

import torch_geometric.llm.models.txt2kg as txt2kg
from torch_geometric.llm.models.txt2kg import (
    TXT2KG,
    _chunk_text,
    _merge_triples_deterministically,
    _multiproc_helper,
    _parse_n_check_triples,
)


def test_init_local_lm_flag():
    model = TXT2KG(local_LM=True, chunk_size=20)
    assert model.local_LM is True
    assert model.initd_LM is False


def test_parse_n_check_triples_formats():
    s = "(A, rel, B)\n(C, rel2, D)"
    parsed = _parse_n_check_triples(s)
    assert ("A", "rel", "B") in parsed
    assert ("C", "rel2", "D") in parsed


def test_chunk_text_simple_sentence():
    text = "Hello world. Another sentence!"
    chunks = _chunk_text(text, chunk_size=10)
    # Only makes chunks at sentence boundaries
    assert any("Hello" in c for c in chunks)


class DummyLLM:
    def __init__(self):
        pass

    def inference(self, *args, **kwargs):
        return ["(X,edge,Y)"]


def test_local_lm_integration(monkeypatch):
    model = TXT2KG(local_LM=True)

    model.model = DummyLLM()
    model.initd_LM = True

    # Simulate time progression
    times = iter([100.0, 100.05])  # 0.05 sec elapsed
    monkeypatch.setattr("time.time", lambda: next(times))

    out = model._chunk_to_triples_str_local("text")

    assert out == "(X,edge,Y)"
    assert model.time_to_parse > 0


def test_add_doc_empty(monkeypatch):
    model = TXT2KG(local_LM=True)
    model.add_doc_2_KG("", QA_pair=None)
    assert model.relevant_triples[0] == []


# Mock LLM + parsing on real text:
def test_add_doc_to_KG(monkeypatch):
    model = TXT2KG(local_LM=True, chunk_size=10)

    # Mock only the LLM output stage
    monkeypatch.setattr(model, "_chunk_to_triples_str_local",
                        lambda *_: "(A,rel,B)\n(C,rel,D)")

    model.add_doc_2_KG("Some text")

    triples = model.relevant_triples[0]

    assert len(triples) == 2
    assert ("A", "rel", "B") in triples
    assert model.doc_id_counter == 1


def test_merge_triples_deterministically_basic():
    # Simple case: multiple sublists, strings only
    results = [
        [["b", "rel", "c"], ["A", "rel", "d"]],
        [["a", "rel", "c"]],
    ]

    merged = _merge_triples_deterministically(results)

    # Expect deterministic, casefolded lexicographic order
    expected = [
        ("a", "rel", "c"),
        ("A", "rel", "d"),
        ("b", "rel", "c"),
    ]
    assert merged == expected


def test_merge_triples_deterministically_unicode_and_nonstring():
    # Include unicode and a numeric element to cover else branch in lambda
    results = [
        [["ä", 2, "x"], ["A", 1, "y"]],
        [["a", 3, "z"]],
    ]

    merged = _merge_triples_deterministically(results)

    # Ensure tuples, unicode sorted, numeric untouched
    expected = [
        ("A", 1, "y"),
        ("a", 3, "z"),
        ("ä", 2, "x"),
    ]
    assert merged == expected


def test_merge_triples_deterministically_empty():
    # Edge case: empty input
    results = []

    merged = _merge_triples_deterministically(results)
    assert merged == []


def test_merge_triples_deterministically_singleton():
    # Edge case: single sublist, single triple
    results = [[["only", "one", "triple"]]]

    merged = _merge_triples_deterministically(results)
    assert merged == [("only", "one", "triple")]


def test_chunk_to_triples_str_cloud(monkeypatch):
    # Fake streaming chunk object
    class DummyChunk:
        class Choice:
            class Delta:
                content = "A"

            delta = Delta()

        choices = [Choice()]

    class DummyCompletion:
        def __iter__(self):
            return iter([DummyChunk()])

    class DummyClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    return DummyCompletion()

            completions = Completions()

        chat = Chat()

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        chat = DummyClient.chat

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = DummyOpenAI

    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    txt2kg.CLIENT_INITD = False

    out = txt2kg._chunk_to_triples_str_cloud("text")
    assert isinstance(out, str)


def dummy_multiproc_helper(
    rank,
    chunks,
    py_fn,
    llm_fn,
    NIM_KEY,
    NIM_MODEL,
    ENDPOINT_URL,
    max_retries=3,
    base_delay=0,
):
    return [("A", "rel", "B")]


def test_extract_relevant_triples_cloud(monkeypatch):

    model = TXT2KG(local_LM=False, chunk_size=10)

    # Mock the multiproc helper (module-level)
    monkeypatch.setattr(txt2kg, "_multiproc_helper", dummy_multiproc_helper)

    triples = model._extract_relevant_triples("Some text")
    assert ("A", "rel", "B") in triples


def test_multiproc_helper_success(monkeypatch):
    # Dummy LLM/Python parser
    def dummy_llm_fn(x, **kwargs):
        return ["llm:" + str(x)]

    def dummy_py_fn(x):
        return ["py:" + str(i) for i in x]

    # Patch _llm_then_python_parse
    monkeypatch.setattr(
        "torch_geometric.llm.models.txt2kg._llm_then_python_parse",
        lambda chunks, py_fn, llm_fn, **kwargs: ["PARSED:" + str(chunks)])

    # Input chunks for rank 0
    chunks_for_rank = ["chunk0", "chunk1"]

    result = _multiproc_helper(
        rank=0,
        chunks_for_rank=chunks_for_rank,
        py_fn=dummy_py_fn,
        llm_fn=dummy_llm_fn,
        NIM_KEY="dummy",
        NIM_MODEL="dummy",
        ENDPOINT_URL="dummy",
        max_retries=3,
        base_delay=0.01  # keep backoff small in tests
    )

    assert result == ["PARSED:['chunk0', 'chunk1']"]


def test_multiproc_helper_retry(monkeypatch):
    attempts = []

    def failing_parse(chunks, py_fn, llm_fn, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("fail")
        return ["SUCCESS"]

    monkeypatch.setattr(
        "torch_geometric.llm.models.txt2kg._llm_then_python_parse",
        failing_parse)

    result = _multiproc_helper(
        rank=0,
        chunks_for_rank=["chunk"],
        py_fn=lambda x: x,
        llm_fn=lambda x: x,
        NIM_KEY="dummy",
        NIM_MODEL="dummy",
        ENDPOINT_URL="dummy",
        max_retries=5,
        base_delay=0  # instant retries for test
    )

    assert result == ["SUCCESS"]
    assert len(attempts) == 3  # retried twice, succeeded on 3rd


def test_add_doc_empty_text():
    kg = TXT2KG(local_LM=True)

    kg.add_doc_2_KG(txt="")

    # first doc uses doc_id_counter=0 as key
    assert 0 in kg.relevant_triples
    assert kg.relevant_triples[0] == []

    # doc counter should increment
    assert kg.doc_id_counter == 1


def test_add_doc_empty_text_with_QA_pair():
    kg = TXT2KG(local_LM=True)

    qa = ("What is PyG?", "Graph ML library")

    kg.add_doc_2_KG(txt="", QA_pair=qa)

    assert qa in kg.relevant_triples
    assert kg.relevant_triples[qa] == []
