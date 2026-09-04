"""Unit tests for Neo4jFeatureStore — mocked driver, no real Neo4j."""
import pickle
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.testing import withPackage


def _make_store(attr_map, default_node_label="Paper", cache=None):
    from examples.neo4j.data.neo4j_feature_store import Neo4jFeatureStore
    return Neo4jFeatureStore(
        attr_map=attr_map,
        uri="bolt://fake:7687",
        user="neo4j",
        pwd="password",
        database_name="neo4j",
        nodeid_property="nodeId",
        default_node_label=default_node_label,
        cache=cache,
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
def _mock_session(records: list):
    """Patch GraphDatabase.driver so session.run iterates *records*."""
    mock_session = MagicMock()
    mock_session.run.return_value = iter(records)
    with _patch_driver(mock_session):
        yield mock_session


@withPackage('neo4j')
def test_attr_map_flat_wrapped_under_default_label():
    store = _make_store({"x": {
        "property": "x",
        "dtype": "float32"
    }}, default_node_label="Paper")
    assert "Paper" in store.attr_map
    assert "x" in store.attr_map["Paper"]


@withPackage('neo4j')
def test_attr_map_nested_kept_as_is():
    nested = {
        "Author": {
            "name": {
                "property": "name",
                "dtype": "str"
            }
        },
        "Paper": {
            "x": {
                "property": "x",
                "dtype": "float32"
            }
        },
    }
    store = _make_store(nested)
    assert store.attr_map["Author"]["name"]["dtype"] == "str"
    assert store.attr_map["Paper"]["x"]["property"] == "x"


@withPackage('neo4j')
def test_attr_map_empty_raises():
    with pytest.raises(ValueError):
        _make_store({})


@withPackage('neo4j')
def test_resolve_group_returns_default_label_when_none():
    store = _make_store({"x": {"property": "x", "dtype": "float32"}})
    resolved = store._resolve_group(None)
    assert resolved == "Paper"


@withPackage('neo4j')
def test_resolve_group_unknown_raises():
    store = _make_store(
        {"Paper": {
            "x": {
                "property": "x",
                "dtype": "float32"
            }
        }},
        default_node_label=None,
    )
    with pytest.raises(KeyError):
        store._resolve_group("DoesNotExist")


@withPackage('neo4j')
def test_build_query_includes_label_and_return_cols():
    store = _make_store({
        "x": {
            "property": "feat",
            "dtype": "float32"
        },
        "y": {
            "property": "label",
            "dtype": "int64"
        },
    })
    query = store._build_query(["x"], "Paper", "Paper")
    assert ":Paper" in query
    assert "n.feat AS x" in query
    assert "UNWIND $node_ids" in query
    assert "nodeId" in query


@withPackage('neo4j')
def test_build_query_no_label_when_none():
    store = _make_store({"x": {
        "property": "x",
        "dtype": "float32"
    }}, default_node_label=None)
    query = store._build_query(["x"], None, None)
    assert ":None" not in query
    assert "MATCH (n {" in query


@withPackage('neo4j')
def test_decode_float_col_f64_array():
    store = _make_store({"x": {"property": "x", "dtype": "float32"}})
    records = [{"x": [1.0, 2.0]}, {"x": [3.0, 4.0]}]
    out = store._decode_float_col(records, "x", "f64[]")
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(1.0)


@withPackage('neo4j')
def test_decode_float_col_byte_encoding():
    store = _make_store(
        {"x": {
            "property": "x",
            "dtype": "float32",
            "encoding": "byte[]"
        }})
    raw = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    records = [{"x": raw}]
    out = store._decode_float_col(records, "x", "byte[]")
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(1.0)


@withPackage('neo4j')
def test_decode_float_col_empty_records():
    store = _make_store({"x": {"property": "x", "dtype": "float32"}})
    out = store._decode_float_col([], "x", "f64[]")
    assert out.shape == (0, )


@withPackage('neo4j')
def test_decode_label_col_int_labels():
    store = _make_store({"y": {"property": "y", "dtype": "int64"}})
    records = [{"y": 3}, {"y": 1}, {"y": 5}]
    out = store._decode_label_col(records, None, "y", string_labels=False)
    assert out.tolist() == [3, 1, 5]


@withPackage('neo4j')
def test_decode_label_col_string_builds_vocab():
    store = _make_store({"y": {"property": "y", "dtype": "str"}})
    records = [{"y": "cat"}, {"y": "dog"}, {"y": "cat"}]
    out = store._decode_label_col(records, "Paper", "y", string_labels=True)
    assert out[0] == out[2]  # same label for "cat"
    assert out[0] != out[1]  # different for "dog"


@withPackage('neo4j')
def test_decode_label_col_none_value_gives_minus_one():
    store = _make_store({"y": {"property": "y", "dtype": "int64"}})
    records = [{"y": None}]
    out = store._decode_label_col(records, None, "y", string_labels=False)
    assert out[0] == -1


@withPackage('neo4j')
def test_put_tensor_db_sends_list_payload_for_f64():
    store = _make_store(
        {"x": {
            "property": "feat",
            "dtype": "float32",
            "encoding": "f64[]"
        }})
    attr = TensorAttr(group_name="Paper", attr_name="x",
                      index=torch.tensor([0], dtype=torch.long))
    tensor = torch.tensor([[1.0, 2.0]])

    with _mock_session([]) as sess:
        store._put_tensor_db(tensor, attr)

    cypher = sess.run.call_args[0][0]
    rows = sess.run.call_args[1]["rows"]
    assert "SET n.feat" in cypher
    assert isinstance(rows[0]["val"], list)


@withPackage('neo4j')
def test_put_tensor_db_sends_bytes_payload_for_byte_encoding():
    store = _make_store(
        {"x": {
            "property": "feat",
            "dtype": "float32",
            "encoding": "byte[]"
        }})
    attr = TensorAttr(group_name="Paper", attr_name="x",
                      index=torch.tensor([0], dtype=torch.long))
    tensor = torch.tensor([[1.0, 2.0]])

    with _mock_session([]) as sess:
        store._put_tensor_db(tensor, attr)

    rows = sess.run.call_args[1]["rows"]
    assert isinstance(rows[0]["val"], bytes)


@withPackage('neo4j')
def test_remove_tensor_db_with_index_uses_unwind():
    store = _make_store({"x": {"property": "feat", "dtype": "float32"}})
    attr = TensorAttr(group_name="Paper", attr_name="x",
                      index=torch.tensor([0, 1], dtype=torch.long))

    with _mock_session([]) as sess:
        store._remove_tensor_db(attr)

    cypher = sess.run.call_args[0][0]
    assert "UNWIND" in cypher
    assert "REMOVE n.feat" in cypher


@withPackage('neo4j')
def test_remove_tensor_db_no_index_wipes_whole_label():
    store = _make_store({"x": {"property": "feat", "dtype": "float32"}})
    attr = TensorAttr(group_name="Paper", attr_name="x", index=None)

    with _mock_session([]) as sess:
        store._remove_tensor_db(attr)

    cypher = sess.run.call_args[0][0]
    assert "UNWIND" not in cypher
    assert "MATCH (n:Paper)" in cypher
    assert "REMOVE n.feat" in cypher


@withPackage('neo4j')
def test_pickle_resets_driver():
    store = _make_store({"x": {"property": "x", "dtype": "float32"}})
    with patch("neo4j.GraphDatabase.driver", return_value=MagicMock()):
        _ = store._get_driver()  # force driver creation
    assert store._driver is not None

    restored = pickle.loads(pickle.dumps(store))
    assert restored._driver is None
    assert restored.attr_map == store.attr_map
    assert restored.nodeid_property == store.nodeid_property
    assert restored.uri == store.uri
    assert restored.default_node_label == store.default_node_label
