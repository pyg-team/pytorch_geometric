import atexit
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from neo4j import Driver, GraphDatabase

from torch_geometric.data.feature_store import (
    DatabaseFeatureStore,
    FeatureCache,
    TensorAttr,
)
from torch_geometric.typing import NodeType


class Neo4jFeatureStore(DatabaseFeatureStore):
    r"""A generalised :class:`~torch_geometric.data.DatabaseFeatureStore`
    backed by Neo4j.

    The set of tensor attributes and their
    Neo4j property mappings are fully configurable at construction time via
    *attr_map*.  The node label used in ``MATCH`` clauses is taken from
    ``TensorAttr.group_name`` at query time, falling back to
    *default_node_label* when ``group_name`` is ``None``.

    A single database round-trip is issued for all attrs that share the same
    node index (the normal mini-batch path), mirroring the optimisation in
    :class:`DatabaseFeatureStore`.

    Args:
        attr_map: Either a flat ``Dict[str, spec]`` (homogeneous: attrs
            live under ``group_name=None`` / ``default_node_label``), or a
            nested ``Dict[Optional[NodeType], Dict[str, spec]]`` keyed by
            node label for heterogeneous graphs. Each *spec* is a plain
            dict with the keys:

            * ``"property"`` *(str)* — Neo4j node property to read.
            * ``"dtype"`` *(str)* — one of ``"float32"``, ``"int64"``,
              or ``"str"`` (string labels encoded to int64 via a
              per-instance vocabulary).
            * ``"encoding"`` *(str, optional)* — ``"f64[]"`` (default) or
              ``"byte[]"``; only used when *dtype* is ``"float32"``.

        uri (str, optional): Bolt URI used to create a driver when *driver*
            is :obj:`None`. (default: :obj:`None`)
        user (str, optional): Username. (default: :obj:`None`)
        pwd (str, optional): Password. (default: :obj:`None`)
        cache (FeatureCache, optional): Per-node-ID feature cache.
            Pass :obj:`None` to disable. (default: :obj:`None`)
        database_name (str, optional): Database to query.
            (default: :obj:`None`)
        nodeid_property (str): Node property used as the integer node ID.
            (default: :obj:`"nodeId"`)
        default_node_label (str, optional): Node label used in the database
            query when ``TensorAttr.group_name`` is ``None``.
    """
    def __init__(
        self,
        attr_map: Dict[Any, Any],
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        cache: Optional[FeatureCache] = None,
        database_name: Optional[str] = None,
        nodeid_property: str = "nodeId",
        default_node_label: Optional[str] = None,
    ) -> None:
        if not attr_map:
            raise ValueError("attr_map must contain at least one entry.")
        super().__init__(cache=cache)
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self.database_name = database_name
        self.nodeid_property = nodeid_property
        self.default_node_label = default_node_label

        # Normalise attr_map to nested {group_name: {attr_name: spec}}. Flat
        # input {attr_name: spec} is wrapped under the single key
        # ``default_node_label`` (or ``None`` when no default is given) so
        # homogeneous callers continue to work unchanged.
        sample_val = next(iter(attr_map.values()))
        is_nested = isinstance(sample_val, dict) and all(
            isinstance(v, dict) for v in attr_map.values()
        ) and not {"property", "dtype", "encoding"} & set(sample_val.keys())
        if is_nested:
            self.attr_map: Dict[Optional[NodeType],
                                Dict[str, Dict[str, str]]] = {
                                    g: dict(spec)
                                    for g, spec in attr_map.items()
                                }
        else:
            self.attr_map = {default_node_label: dict(attr_map)}

        self._labels: Dict[Tuple[Optional[NodeType], str], Dict[str, int]] = {
            (group, name): {}
            for group, group_map in self.attr_map.items()
            for name, spec in group_map.items() if spec.get("dtype") == "str"
        }

    def _resolve_group(self,
                       group_name: Optional[NodeType]) -> Optional[NodeType]:
        """Pick the attr_map group key for *group_name*.

        Tries the literal group, then ``default_node_label``, then ``None``.
        Returns the first key present in :attr:`attr_map`.
        """
        for candidate in (group_name, self.default_node_label, None):
            if candidate in self.attr_map:
                return candidate
        raise KeyError(
            f"No attr_map entry for group '{group_name}' "
            f"(tried default '{self.default_node_label}' and None). "
            f"Known groups: {list(self.attr_map)}")

    def _get_spec(
        self,
        group_name: Optional[NodeType],
        attr_name: str,
    ) -> Dict[str, str]:
        group = self._resolve_group(group_name)
        group_map = self.attr_map[group]
        if attr_name not in group_map:
            raise ValueError(f"attr '{attr_name}' is not registered for group "
                             f"'{group}'. Known attrs: {list(group_map)}")
        return group_map[attr_name]

    def _build_query(
        self,
        attr_names: List[str],
        node_label: Optional[str],
        group_key: Optional[NodeType],
    ) -> str:
        """Build a single Cypher query that returns all requested attrs."""
        lf = f":{node_label}" if node_label else ""
        group_map = self.attr_map[group_key]
        return_cols = ", ".join(f"n.{group_map[name]['property']} AS {name}"
                                for name in attr_names)
        return (f"UNWIND $node_ids AS nid "
                f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
                f"RETURN nid AS id, {return_cols}")

    def _fetch_remote_attrs(
        self,
        attr: TensorAttr,
    ) -> Tuple[List[object], List[int]]:
        """Fetch a single attr from the database."""
        index = attr.index
        node_ids: List[int] = (index.tolist() if isinstance(
            index, torch.Tensor) else list(index))

        group_key = self._resolve_group(attr.group_name)
        node_label = (attr.group_name if attr.group_name is not None else
                      self.default_node_label)
        query = self._build_query([attr.attr_name], node_label, group_key)
        with self._get_driver().session(database=self.database_name,
                                        fetch_size=1000) as session:
            records = list(session.run(query, node_ids=node_ids))

        fetched_nids = [rec["id"] for rec in records]
        return records, fetched_nids

    def _decode_remote_attrs(
        self,
        records: List[object],
        attr: TensorAttr,
    ) -> np.ndarray:
        """Decode a single attr from records."""
        name = attr.attr_name
        spec = self._get_spec(attr.group_name, name)

        if not records:
            if spec.get("dtype", "float32") == "float32":
                return np.empty((0, ), dtype=np.float32)
            else:
                return np.empty((0, ), dtype=np.int64)

        if spec.get("dtype", "float32") == "float32":
            return self._decode_float_col(records, name,
                                          spec.get("encoding", "f64[]"))
        else:
            return self._decode_label_col(records, attr.group_name, name,
                                          spec.get("dtype") == "str")

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
        group_name: Optional[NodeType],
        field: str,
        string_labels: bool,
    ) -> np.ndarray:
        arr = np.empty(len(records), dtype=np.int64)
        group_key = self._resolve_group(group_name) if string_labels else None
        vocab = self._labels.get(
            (group_key, field), {}) if string_labels else {}
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
        if string_labels:
            self._labels[(group_key, field)] = vocab
        return arr

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [
            TensorAttr(group_name=group, attr_name=name)
            for group, group_map in self.attr_map.items() for name in group_map
        ]

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        out = self._get_tensor(attr)
        if out is None:
            return None
        return tuple(out.size())

    def _put_tensor_db(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        r"""Write *tensor* rows back to Neo4j.

        ``attr.index`` identifies which node IDs to update.  The tensor must
        have shape ``(len(index), *feature_shape)`` or scalar shape
        ``(len(index),)``.  The attr must be registered in *attr_map*.
        """
        name = attr.attr_name
        spec = self._get_spec(attr.group_name, name)
        prop = spec["property"]
        encoding = spec.get("encoding", "f64[]")
        dtype = spec.get("dtype", "float32")

        index = attr.index
        node_ids: List[int] = (index.tolist() if isinstance(
            index, torch.Tensor) else list(index))

        arr = tensor.detach().cpu().numpy()

        rows: List[dict] = []
        for i, nid in enumerate(node_ids):
            row = arr[i]
            if dtype == "float32" and encoding == "byte[]":
                val = row.astype(np.float32).tobytes()
            elif dtype == "float32":
                val = row.astype(np.float64).tolist()
            else:
                val = int(row) if row.ndim == 0 else row.tolist()
            rows.append({"nid": int(nid), "val": val})

        group = attr.group_name
        node_label = group if group is not None else self.default_node_label
        lf = f":{node_label}" if node_label else ""
        query = (f"UNWIND $rows AS r "
                 f"MATCH (n{lf} {{{self.nodeid_property}: r.nid}}) "
                 f"SET n.{prop} = r.val")

        with self._get_driver().session(
                database=self.database_name) as session:
            session.run(query, rows=rows)

        return True

    def _remove_tensor_db(self, attr: TensorAttr) -> bool:
        r"""Remove a node property from Neo4j for the nodes in ``attr.index``.

        If ``attr.index`` is ``None`` the property is removed from every node
        of the matching label.
        """
        name = attr.attr_name
        spec = self._get_spec(attr.group_name, name)
        prop = spec["property"]
        group = attr.group_name
        node_label = group if group is not None else self.default_node_label
        lf = f":{node_label}" if node_label else ""

        if attr.index is not None:
            index = attr.index
            node_ids: List[int] = (index.tolist() if isinstance(
                index, torch.Tensor) else list(index))
            query = (f"UNWIND $node_ids AS nid "
                     f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
                     f"REMOVE n.{prop}")
            with self._get_driver().session(
                    database=self.database_name) as session:
                session.run(query, node_ids=node_ids)
        else:
            query = f"MATCH (n{lf}) REMOVE n.{prop}"
            with self._get_driver().session(
                    database=self.database_name) as session:
                session.run(query)

        return True

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
