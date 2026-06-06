"""Unit tests for Neo4jGraphStore — mocked driver, no real Neo4j."""
import pickle
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch
from neo4j_graph_store import Neo4jGraphStore

from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout


def _make_store():
    return Neo4jGraphStore(
        uri="bolt://fake:7687",
        user="neo4j",
        pwd="password",
        nodeid_property="nodeId",
        database_name="neo4j",
    )


@contextmanager
def _patch_driver(mock_session):
    """Patch GraphDatabase.driver to yield *mock_session*."""
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        yield mock_session


@contextmanager
def _mock_session(records):
    mock_session = MagicMock()
    mock_session.run.return_value = iter(records)
    with _patch_driver(mock_session):
        yield mock_session


def test_get_edge_index_coo_returns_row_col():
    store = _make_store()
    records = [{"src": 0, "dst": 1}, {"src": 1, "dst": 2}]
    attr = EdgeAttr(edge_type=("Paper", "CITES", "Paper"),
                    layout=EdgeLayout.COO, is_sorted=False, size=None)
    with _mock_session(records):
        row, col = store._get_edge_index(attr)
    assert row.tolist() == [0, 1]
    assert col.tolist() == [1, 2]


def test_get_edge_index_empty_returns_empty_tensors():
    store = _make_store()
    attr = EdgeAttr(edge_type=("Paper", "CITES", "Paper"),
                    layout=EdgeLayout.COO, is_sorted=False, size=None)
    with _mock_session([]):
        result = store._get_edge_index(attr)
    row, col = result
    assert row.numel() == 0
    assert col.numel() == 0


def test_get_edge_index_csc_sorts_by_col_preserving_pairs():
    store = _make_store()
    records = [{"src": 2, "dst": 0}, {"src": 1, "dst": 1}]
    attr = EdgeAttr(edge_type=("Paper", "CITES", "Paper"),
                    layout=EdgeLayout.CSC, is_sorted=False, size=None)
    with _mock_session(records):
        row, col = store._get_edge_index(attr)
    # CSC: sorted by col, row reordered consistently with its pair.
    assert col.tolist() == [0, 1]
    assert row.tolist() == [2, 1]


def test_put_edge_index_issues_merge_query():
    store = _make_store()
    attr = EdgeAttr(edge_type=("Paper", "CITES", "Paper"),
                    layout=EdgeLayout.COO, is_sorted=False, size=None)
    with _mock_session([]) as sess:
        store._put_edge_index((torch.tensor([0]), torch.tensor([1])), attr)
    cypher = sess.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "CITES" in cypher


def test_remove_edge_index_issues_delete_query():
    store = _make_store()
    attr = EdgeAttr(edge_type=("Paper", "CITES", "Paper"),
                    layout=EdgeLayout.COO, is_sorted=False, size=None)
    with _mock_session([]) as sess:
        store._remove_edge_index(attr)
    cypher = sess.run.call_args[0][0]
    assert "DELETE" in cypher
    assert "CITES" in cypher


def test_apoc_available_true():
    store = _make_store()
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"v": "5.0"}
    with _patch_driver(mock_session):
        assert store.apoc_available() is True


def test_apoc_available_false_on_exception():
    store = _make_store()
    mock_session = MagicMock()
    mock_session.run.side_effect = Exception("not found")
    with _patch_driver(mock_session):
        assert store.apoc_available() is False


def test_pickle_resets_driver():
    store = _make_store()
    with patch("neo4j.GraphDatabase.driver", return_value=MagicMock()):
        _ = store._get_driver()
    assert store._driver is not None

    restored = pickle.loads(pickle.dumps(store))
    assert restored._driver is None
    assert restored.uri == store.uri
    assert restored.nodeid_property == store.nodeid_property
    assert restored.database_name == store.database_name
