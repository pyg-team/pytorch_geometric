"""Neo4j-backed GraphSAINT samplers.

Provides a concrete :class:`Neo4jGraphSAINTSampler` base and a ready-to-use
:class:`Neo4jGraphSAINTRandomWalkSampler` variant.
"""
# isort: skip_file
from typing import Any, Dict, List, Optional, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.loader.database_graph_saint import (
    DatabaseGraphSAINTSampler)
from examples.neo4j.data.Neo4jFeatureStore import Neo4jFeatureStore
from examples.neo4j.data.Neo4jGraphStore import Neo4jGraphStore


class Neo4jGraphSAINTSampler(DatabaseGraphSAINTSampler):
    """Neo4j-backed GraphSAINT base.

    Implements the three database-specific hooks shared by all Neo4j variants:
    :meth:`_get_total_nodes`, :meth:`_build_subgraph_query`, and
    :meth:`_build_data`.

    Feature and label retrieval in :meth:`_build_data` goes through
    :meth:`~torch_geometric.data.DatabaseFeatureStore.multi_get_tensor` so that
    caching and encoding are handled by :class:`Neo4jFeatureStore`.

    Args:
        graph_store (Neo4jGraphStore): Neo4j graph store.
        feature_store (Neo4jFeatureStore): Neo4j feature store.
        feature_attrs (dict, optional): Mapping from
            :class:`~torch_geometric.data.Data` attribute name to a
            ``(store_attr_name, dtype)`` tuple.  Each entry produces one
            :class:`~torch_geometric.data.TensorAttr` fetch and is
            assigned to the corresponding key on the returned
            :class:`~torch_geometric.data.Data` object.
            (default: ``{"x": ("x", torch.float), "y": ("y", torch.long)}``)
        node_label (str, optional): Neo4j node label used in ``MATCH``
            clauses.  ``None`` matches all labels.
            (default: :obj:`None`)
        rel_type (str, optional): Relationship type filter.
            ``None`` matches all types. (default: :obj:`None`)
        split_property (str, optional): Node property storing the split
            assignment (e.g. ``"split"``).  When set the subgraph query
            returns it and :meth:`_build_data` attaches a ``train_mask``.
            (default: :obj:`None`)
        train_split_value: The value of ``split_property`` that identifies
            training nodes.  Accepts a string (e.g. ``"train"``) or an
            integer (e.g. ``0`` when splits are encoded as 0/1/2).
            (default: :obj:`"train"`)
        batch_size (int): Passed through to the base class.
        num_steps (int): Passed through to the base class.
        sample_coverage (int): Passed through to the base class.
            (default: :obj:`50`)
        save_dir (str, optional): Passed through to the base class.
    """
    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        feature_store: Neo4jFeatureStore,
        feature_attrs: Optional[Dict[str, Tuple[str, torch.dtype]]] = None,
        node_label: Optional[str] = None,
        rel_type: Optional[str] = None,
        split_property: Optional[str] = None,
        train_split_value: Any = "train",
        batch_size: int = 1,
        num_steps: int = 1,
        sample_coverage: int = 50,
        save_dir: Optional[str] = None,
    ) -> None:
        # Set Neo4j-specific attrs before super().__init__ because
        # _build_subgraph_query() is called inside the base __init__.
        if feature_attrs is None:
            feature_attrs = {
                "x": ("x", torch.float),
                "y": ("y", torch.long),
            }
        self.feature_attrs = feature_attrs
        self.node_label = node_label
        self.rel_type = rel_type
        self.split_property = split_property
        self.train_split_value = train_split_value
        self.nodeid_property = graph_store.nodeid_property

        super().__init__(
            graph_store=graph_store,
            feature_store=feature_store,
            batch_size=batch_size,
            num_steps=num_steps,
            sample_coverage=sample_coverage,
            save_dir=save_dir,
        )

    def _get_total_nodes(self) -> int:
        lbl = f":{self.node_label}" if self.node_label else ""
        query = f"MATCH (n{lbl}) RETURN count(n) AS total"
        with self.graph_store._get_driver().session(
                database=self.graph_store.database_name) as session:
            record = session.run(query).single()
        return int(record["total"]) if record else 0

    def _build_subgraph_query(self) -> str:
        """Induced-subgraph query for ``$nodes`` (list of global node IDs).

        Returns ``nodes``, ``edges``, and optionally ``splits`` in one
        round-trip.
        """
        nid = self.nodeid_property
        node_lbl = f":{self.node_label}" if self.node_label else ""
        rel_flt = f":{self.rel_type}" if self.rel_type else ""

        split_col = (f", [n IN nodes | n.{self.split_property}] AS splits"
                     if self.split_property else "")

        return f"""
        MATCH (n{node_lbl}) WHERE n.{nid} IN $nodes
        WITH collect(n) AS nodes
        CALL (nodes) {{
            UNWIND nodes AS a
            OPTIONAL MATCH (a)-[r{rel_flt}]->(b) WHERE b IN nodes
            RETURN collect(
                CASE WHEN b IS NOT NULL THEN [a.{nid}, b.{nid}] END
            ) AS edges
        }}
        RETURN
            [n IN nodes | n.{nid}] AS nodes,
            edges{split_col}
        """

    def _build_data(
        self,
        record: dict,
        nodes: List[int],
        edge_index: torch.Tensor,
        node_norm: Optional[torch.Tensor],
    ) -> Data:
        if not nodes:
            return Data(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=0,
            )

        idx = torch.tensor(nodes, dtype=torch.long)

        # Build TensorAttr list from user-defined feature_attrs.
        data_keys: List[str] = []
        tensor_attrs: List[TensorAttr] = []
        dtypes: List[torch.dtype] = []
        for data_key, (attr_name, dtype) in self.feature_attrs.items():
            data_keys.append(data_key)
            tensor_attrs.append(
                TensorAttr(group_name=self.node_label, attr_name=attr_name,
                           index=idx), )
            dtypes.append(dtype)

        tensors = self.feature_store.multi_get_tensor(tensor_attrs)

        data = Data(edge_index=edge_index, num_nodes=len(nodes))
        for key, tensor, dtype in zip(data_keys, tensors, dtypes):
            data[key] = tensor.to(dtype)

        # Optional train mask from the split property.
        # Use .get() for compatibility with neo4j Record objects.
        splits_raw = record.get("splits") if record else None
        if self.split_property and splits_raw is not None:
            data.train_mask = torch.tensor(
                [s == self.train_split_value for s in splits_raw],
                dtype=torch.bool,
            )

        if node_norm is not None:
            data.node_norm = node_norm

        return data

    # _sample_nodes intentionally left abstract — filled by variant subclasses.
    def _sample_nodes(self) -> List[int]:
        raise NotImplementedError


class Neo4jGraphSAINTRandomWalkSampler(Neo4jGraphSAINTSampler):
    """GraphSAINT Random Walk Sampler backed by Neo4j.

    Each subgraph is formed by running ``batch_size`` random walks of
    ``walk_length`` steps from randomly chosen root nodes, then taking the
    induced subgraph on all visited node IDs.

    Args:
        graph_store (Neo4jGraphStore): Neo4j graph store.
        feature_store (Neo4jFeatureStore): Neo4j feature store.
        batch_size (int): Number of random-walk root nodes per subgraph.
        walk_length (int): Steps per random walk.
        num_steps (int): Subgraphs per epoch. (default: :obj:`1`)
        **kwargs: Remaining args forwarded to :class:`Neo4jGraphSAINTSampler`.
    """
    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        feature_store: Neo4jFeatureStore,
        batch_size: int,
        walk_length: int,
        num_steps: int = 1,
        **kwargs,
    ) -> None:
        # Must be set before super().__init__ because _setup() (called inside
        # the base __init__) uses walk_length to build _walk_query.
        self.walk_length = walk_length
        super().__init__(
            graph_store=graph_store,
            feature_store=feature_store,
            batch_size=batch_size,
            num_steps=num_steps,
            **kwargs,
        )

    @property
    def _filename(self) -> str:
        return (f"neo4j_graphsaint_rw"
                f"_{self.walk_length}_{self.sample_coverage}.pt")

    def _setup(self) -> None:
        self._walk_query = self._build_walk_query()

    def _sample_nodes(self) -> List[int]:
        record = self.graph_store._fetch_subgraph(
            self._walk_query, {"batch_size": self._batch_size})
        if record is None or not record.get("nodes"):
            return []
        return list(record["nodes"])

    def _build_walk_query(self) -> str:
        """Cypher random-walk query using APOC.

        Picks ``$batch_size`` random roots, unrolls ``walk_length`` steps
        each as a ``CALL`` block, and returns deduplicated visited node IDs.
        """
        nid = self.nodeid_property
        node_lbl = f":{self.node_label}" if self.node_label else ""
        rel_flt = f":{self.rel_type}" if self.rel_type else ""

        q: List[str] = []

        q.append(f"""
        MATCH (root{node_lbl})
        WITH collect(root) AS all_nodes
        WITH apoc.coll.randomItems(all_nodes, $batch_size, false) AS roots
        WITH roots AS walkers,
             [n IN roots | n.{nid}] AS visited_ids
        """)

        for step in range(self.walk_length):
            q.append(f"""
        // step {step + 1}
        CALL (walkers) {{
            UNWIND walkers AS walker
            OPTIONAL MATCH (walker)<-[r{rel_flt}]-(nbr{node_lbl})
            WITH walker, collect(nbr) AS candidates
            WITH CASE WHEN size(candidates) > 0
                      THEN candidates[toInteger(rand() * size(candidates))]
                      ELSE walker
                 END AS next_pos
            RETURN collect(next_pos) AS new_walkers
        }}
        WITH new_walkers AS walkers,
             visited_ids + [n IN new_walkers | n.{nid}] AS visited_ids
        """)

        q.append("RETURN apoc.coll.toSet(visited_ids) AS nodes")
        return "\n".join(q)
