import atexit
from typing import List, Optional, Tuple

import torch
from neo4j import Driver, GraphDatabase
from torch import Tensor

from torch_geometric.data.graph_store import EdgeAttr, RemoteGraphStore
from torch_geometric.typing import EdgeTensorType


class Neo4jGraphStore(RemoteGraphStore):
    """Lean Neo4j-backed graph store with no benchmarking or profiling.

    Implements all abstract methods of :class:`RemoteGraphStore` against a
    Neo4j database.  The driver is created lazily per process, making this
    safe to use with multi-process DataLoader workers (``num_workers > 0``).

    Sampling queries are **not** defined here — that is the responsibility of
    the sampler (e.g. :class:`~neo4j_pyg.samplers.Neo4jSampler`).  Use
    :meth:`sample_from_nodes` to execute an arbitrary Cypher query and
    :meth:`_fetch_subgraph` / :meth:`_decode_subgraph` to execute the query
    and convert the result into PyG tensors.

    Args:
        uri (str): Bolt URI of the Neo4j instance.
        user (str): Database username.
        pwd (str): Database password.
        database_name (str, optional): Database name. Defaults to
            ``dataset_name``.
        dataset_name (str): Logical dataset name used as the default database.
            (default: ``"neo4j"``)
        split_property_name (str): Node property that encodes the
            train/val/test split. (default: ``"split"``)
        split_property_type (str): ``"int"`` or ``"str"`` — how the split
            value is stored. (default: ``"int"``)
        nodeid_property (str): Node property used as the global node ID.
            (default: ``"nodeId"``)
    """
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        database_name: Optional[str] = None,
        dataset_name: str = "neo4j",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
    ):
        super().__init__()
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self.database_name = database_name if database_name else dataset_name
        self.dataset_name = dataset_name
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
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

        if record is None or (not record.get("nodes")
                              and not record.get("nodes_by_hop")):
            return seed_nodes, empty, empty

        if record.get("nodes"):
            global_ids = torch.tensor(record["nodes"], dtype=torch.long)
        else:
            flat = [nid for hop in record["nodes_by_hop"] for nid in hop]
            global_ids = torch.tensor(flat, dtype=torch.long)

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
        r"""Remote stores do not support full edge index retrieval by default.

        Sampling is expected to go through :meth:`_fetch_subgraph` /
        :meth:`_decode_subgraph` and a compatible
        :class:`~torch_geometric.sampler.BaseSampler` instead.  Override this
        method if your backend can efficiently return the entire edge index.
        """
        raise NotImplementedError(
            "Remote stores do not support full edge index retrieval.")

    def _put_edge_index(
        self,
        edge_index: EdgeTensorType,
        edge_attr: EdgeAttr,
    ) -> bool:
        raise NotImplementedError(
            "Remote stores do not support full edge index retrieval.")

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        raise NotImplementedError(
            "Remote stores do not support full edge index retrieval.")
