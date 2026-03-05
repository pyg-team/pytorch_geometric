from torch_geometric.llm.models.txt2kg import (
     TXT2KG,
     _parse_n_check_triples,
     _chunk_text,
     _merge_triples_deterministically,
)
from torch_geometric.testing import onlyRAG


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
    def __init__(self): pass
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
    monkeypatch.setattr(
        model,
        "_chunk_to_triples_str_local",
        lambda *_: "(A,rel,B)\n(C,rel,D)"
    )

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
