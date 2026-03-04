from torch_geometric.testing import onlyRAG
from torch_geometric.llm.models.txt2kg import _merge_triples_deterministically


@onlyRAG
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


@onlyRAG
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


@onlyRAG
def test_merge_triples_deterministically_empty():
    # Edge case: empty input
    results = []

    merged = _merge_triples_deterministically(results)
    assert merged == []


@onlyRAG
def test_merge_triples_deterministically_singleton():
    # Edge case: single sublist, single triple
    results = [[["only", "one", "triple"]]]

    merged = _merge_triples_deterministically(results)
    assert merged == [("only", "one", "triple")]

