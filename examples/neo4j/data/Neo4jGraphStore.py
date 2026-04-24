import atexit
from typing import List, Optional, Tuple

import torch
from neo4j import Driver, GraphDatabase
from torch import Tensor

from torch_geometric.data.graph_store import (
    DatabaseGraphStore,
    EdgeAttr,
    EdgeLayout,
)
from torch_geometric.typing import EdgeTensorType


class Neo4jGraphStore(DatabaseGraphStore):
    """Neo4j-backed graph store.

    Implements all abstract methods of :class:`DatabaseGraphStore` against a
    Neo4j database.  The driver is created lazily per process, making this
    safe to use with multi-process DataLoader workers (``num_workers > 0``).

    Args:
        uri (str): Bolt URI of the Neo4j instance.
        user (str): Username.
        pwd (str): Password.
        database_name (str): Database name.
        nodeid_property (str): Property used as the global node ID.
    """
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        nodeid_property: str,
        database_name: Optional[str] = None,
    ):
        super().__init__()
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self.database_name = database_name
        self.nodeid_property = nodeid_property

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri,
                                                auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver

    def close(self) -> None:
        if getattr(self, "_driver", None) is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_driver"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None

    def _decode_subgraph(
        self,
        record: Optional[dict],
        seed_nodes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert a raw ``sample_from_nodes`` record into
        ``(node, row, col)`` COO tensors.

        On an empty result *seed_nodes* are returned as the node tensor with
        zero-length edge tensors.
        """
        empty = torch.zeros(0, dtype=torch.long)

        if record is None or not record.get("nodes", None):
            return seed_nodes, empty, empty

        global_ids = torch.tensor(record["nodes"], dtype=torch.long)

        global_to_local = {
            int(gid): i
            for i, gid in enumerate(global_ids.tolist())
        }

        edges = record.get("edges") or []
        if edges:
            row = torch.tensor([global_to_local[e[0]] for e in edges],
                               dtype=torch.long)
            col = torch.tensor([global_to_local[e[1]] for e in edges],
                               dtype=torch.long)
        else:
            row = col = empty

        return global_ids, row, col

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return []

    def _fetch_subgraph(self, query: str, kwargs: dict) -> Optional[dict]:
        """Execute *query* against the DB and return the first result record.

        Returns the record dict, or ``None`` if the query produced no rows.
        """
        with self._get_driver().session(database=self.database_name,
                                        fetch_size=-1) as session:
            result = session.run(query, **kwargs)
            records = list(result)
        return records[0] if records else None

    # GraphStore ABC ##########################################################
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        src_type, rel_type, dst_type = edge_attr.edge_type
        nid = self.nodeid_property

        query = (f"MATCH (a:{src_type})-[r:{rel_type}]->(b:{dst_type}) "
                 f"RETURN a.{nid} AS src, b.{nid} AS dst")
        with self._get_driver().session(database=self.database_name,
                                        fetch_size=-1) as session:
            records = list(session.run(query))

        if not records:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty

        row = torch.tensor([rec["src"] for rec in records], dtype=torch.long)
        col = torch.tensor([rec["dst"] for rec in records], dtype=torch.long)

        if edge_attr.layout == EdgeLayout.CSC:
            col, perm = col.sort()
            row = row[perm]
        elif edge_attr.layout == EdgeLayout.CSR:
            row, perm = row.sort()
            col = col[perm]

        return row, col

    def _put_edge_index(
        self,
        edge_index: EdgeTensorType,
        edge_attr: EdgeAttr,
    ) -> bool:
        """Write edges to Neo4j via ``MERGE``.

        Values in *edge_index* must be global node IDs stored under
        ``nodeid_property``.
        """
        src_type, rel_type, dst_type = edge_attr.edge_type
        if isinstance(edge_index, tuple):
            rows, cols = edge_index
        else:
            rows, cols = edge_index[0], edge_index[1]
        pairs = list(zip(rows.tolist(), cols.tolist()))
        nid = self.nodeid_property
        query = (f"UNWIND $pairs AS p "
                 f"MATCH (a:{src_type} {{{nid}: p[0]}}) "
                 f"MATCH (b:{dst_type} {{{nid}: p[1]}}) "
                 f"MERGE (a)-[:{rel_type}]->(b)")
        with self._get_driver().session(
                database=self.database_name) as session:
            session.run(query, pairs=pairs)
        return True

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        """Delete all relationships of the given edge type from Neo4j."""
        src_type, rel_type, dst_type = edge_attr.edge_type
        query = (f"MATCH (:{src_type})-[r:{rel_type}]->(:{dst_type}) "
                 f"DELETE r")
        with self._get_driver().session(
                database=self.database_name) as session:
            session.run(query)
        return True
