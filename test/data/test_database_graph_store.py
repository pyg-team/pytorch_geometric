"""Unit tests for DatabaseGraphStore plumbing."""
import pytest

from torch_geometric.data.database_graph_store import DatabaseGraphStore
from torch_geometric.testing import FakeDatabaseGraphStore


def test_query_db_passes_query_and_kwargs():
    store = FakeDatabaseGraphStore(records={"Q": {"nodes": [1], "edges": []}})
    store.query_db("Q", {"foo": "bar"})
    assert store.executed_queries[-1] == ("Q", {"foo": "bar"})


def test_query_db_return_value_propagates():
    record = {"nodes": [1, 2], "edges": [[1, 2]]}
    store = FakeDatabaseGraphStore(records={"Q": record})
    assert store.query_db("Q", {}) is record


def test_abstract_without_query_db_raises():
    class Incomplete(DatabaseGraphStore):
        pass

    with pytest.raises(TypeError, match="query_db"):
        Incomplete()
