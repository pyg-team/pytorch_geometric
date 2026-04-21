import atexit
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from neo4j import Driver, GraphDatabase

from torch_geometric.data.feature_store import RemoteFeatureStore, TensorAttr


class Neo4jFeatureStore(RemoteFeatureStore):
    r"""A generalised :class:`~torch_geometric.data.RemoteFeatureStore` backed
    by Neo4j.

    Unlike :class:`Neo4jFeatureStore`, the set of tensor attributes and their
    Neo4j property mappings are fully configurable at construction time via
    *attr_map*.  The node label used in ``MATCH`` clauses is taken from
    ``TensorAttr.group_name`` at query time, falling back to
    *default_node_label* when ``group_name`` is ``None``.

    A single Cypher round-trip is issued for all attrs that share the same
    node index (the normal mini-batch path), mirroring the optimisation in
    :class:`Neo4jFeatureStore`.

    Args:
        attr_map (Dict[str, dict]): Mapping from logical attr name (e.g.
            ``"x"``, ``"y"``) to a plain dict with the keys:

            * ``"property"`` *(str)* — Neo4j node property to read.
            * ``"dtype"`` *(str)* — one of ``"float32"``, ``"int64"``,
              or ``"str"`` (string labels encoded to int64 via a
              per-instance vocabulary).
            * ``"encoding"`` *(str, optional)* — ``"f64[]"`` (default) or
              ``"byte[]"``; only used when *dtype* is ``"float32"``.

        driver (neo4j.Driver, optional): An already-open driver.  When
            provided *uri* / *user* / *pwd* are ignored and the driver is
            never closed by this class. (default: :obj:`None`)
        uri (str, optional): Bolt URI used to create a driver when *driver*
            is :obj:`None`. (default: :obj:`None`)
        user (str, optional): Neo4j username. (default: :obj:`None`)
        pwd (str, optional): Neo4j password. (default: :obj:`None`)
        cache (MutableMapping, optional): Per-node-ID feature cache.  Pass
            :obj:`None` to disable. (default: :obj:`None`)
        database_name (str, optional): Neo4j database to query.  Defaults to
            *dataset_name*. (default: :obj:`None`)
        dataset_name (str): Fallback database name.
            (default: :obj:`"neo4j"`)
        nodeid_property (str): Node property used as the integer node ID.
            (default: :obj:`"nodeId"`)
        default_node_label (str, optional): Neo4j node label used in
            ``MATCH`` when ``TensorAttr.group_name`` is ``None``.
            (default: :obj:`None`)

    Example::

        store = Neo4jFeatureStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            pwd="password",
            attr_map={
                "x": {"property": "features", "dtype": "float32",
                      "encoding": "f64[]"},
                "y": {"property": "category", "dtype": "str"},
            },
        )
    """
    def __init__(
        self,
        attr_map: Dict[str, Dict[str, str]],
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        cache: Optional[MutableMapping] = None,
        database_name: Optional[str] = None,
        dataset_name: str = "neo4j",
        nodeid_property: str = "nodeId",
        default_node_label: Optional[str] = None,
    ) -> None:
        super().__init__(cache=cache)
        if not attr_map:
            raise ValueError("attr_map must contain at least one entry.")
        self.attr_map = attr_map
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self.database_name = database_name if database_name else dataset_name
        self.dataset_name = dataset_name
        self.nodeid_property = nodeid_property
        self.default_node_label = default_node_label
        # Per-attr string-label vocabularies (populated lazily during decoding)
        self._labels: Dict[str, Dict[str, int]] = {
            name: {}
            for name, spec in attr_map.items() if spec.get("dtype") == "str"
        }

    def _build_query(
        self,
        attr_names: List[str],
        node_label: Optional[str],
    ) -> str:
        """Build a single Cypher query that returns all requested attrs."""
        lf = f":{node_label}" if node_label else ""
        return_cols = ", ".join(
            f"n.{self.attr_map[name]['property']} AS {name}"
            for name in attr_names)
        return (f"UNWIND $node_ids AS nid "
                f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
                f"RETURN nid AS id, {return_cols}")

    def _fetch_remote_attrs(
        self,
        node_ids: List[int],
        attrs: List[TensorAttr],
    ) -> Tuple[List[object], List[int]]:
        attr_names = [a.attr_name for a in attrs]
        group = attrs[0].group_name if attrs else None
        node_label = group if group is not None else self.default_node_label

        query = self._build_query(attr_names, node_label)
        with self._get_driver().session(database=self.database_name,
                                        fetch_size=1000) as session:
            records = list(session.run(query, node_ids=node_ids))

        fetched_nids = [rec["id"] for rec in records]
        return records, fetched_nids

    def _decode_remote_attrs(
        self,
        records: Any,
        attrs: List[TensorAttr],
    ) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}

        for attr in attrs:
            name = attr.attr_name
            spec = self.attr_map[name]

            if not records:
                if spec.get("dtype", "float32") == "float32":
                    result[name] = np.empty((0, ), dtype=np.float32)
                else:
                    result[name] = np.empty((0, ), dtype=np.int64)
                continue

            if spec.get("dtype", "float32") == "float32":
                result[name] = self._decode_float_col(
                    records, name, spec.get("encoding", "f64[]"))
            else:
                result[name] = self._decode_label_col(
                    records, name,
                    spec.get("dtype") == "str")

        return result

    def _decode_float_col(
        self,
        records: List[object],
        field: str,
        encoding: str,
    ) -> np.ndarray:
        if encoding == "byte[]":
            return np.stack([
                np.frombuffer(memoryview(rec[field]), dtype=np.float32)
                for rec in records
            ])
        return np.asarray([rec[field] for rec in records], dtype=np.float32)

    def _decode_label_col(
        self,
        records: List[object],
        field: str,
        string_labels: bool,
    ) -> np.ndarray:
        arr = np.empty(len(records), dtype=np.int64)
        vocab = self._labels.get(field, {})
        for i, rec in enumerate(records):
            val = rec[field]
            if val is None:
                arr[i] = -1
            elif string_labels:
                if val not in vocab:
                    vocab[val] = len(vocab)
                arr[i] = vocab[val]
            else:
                arr[i] = int(val)
        return arr

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [
            TensorAttr(group_name=None, attr_name=name)
            for name in self.attr_map
        ]

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        out = self._get_tensor(attr)
        if out is None:
            return None
        return tuple(out.size())

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        raise NotImplementedError(
            "Tensors should be fetched via _fetch_remote_attrs"
            "to avoid multiple round trips to the database.")

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise NotImplementedError(
            "Tensors should be fetched via _fetch_remote_attrs"
            "to avoid multiple round trips to the database.")

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri,
                                                auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver

    def close(self) -> None:
        r"""Close the internally-managed driver, if any."""
        if getattr(self, "_driver", None) is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_driver"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._driver = None
