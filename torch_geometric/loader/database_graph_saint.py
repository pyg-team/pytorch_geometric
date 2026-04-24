r"""Remote-backed GraphSAINT sampler base class.

Mirrors :class:`~torch_geometric.loader.GraphSAINTSampler` but replaces the
in-memory graph with a :class:`~torch_geometric.data.DatabaseGraphStore` +
:class:`~torch_geometric.data.RemoteFeatureStore` pair so that subgraph
sampling and feature retrieval happen via remote round-trips rather than
local tensor indexing.

Normalization follows PyG's formula exactly:
``node_norm[v] = num_samples / visit_count[v] / N``.
Nodes never visited during pre-sampling get ``count = 0.1`` (same as PyG).
"""
import os.path as osp
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.data.feature_store import DatabaseFeatureStore
from torch_geometric.data.graph_store import DatabaseGraphStore


class DatabaseGraphSAINTSampler(torch.utils.data.DataLoader):
    r"""Abstract base class for database-backed GraphSAINT samplers.

    Provides the DataLoader scaffolding and norm pre-computation.  Concrete
    subclasses decide *how* to sample nodes (:meth:`_sample_nodes`), *what*
    the subgraph query looks like (:meth:`_build_subgraph_query`), and *how*
    to assemble the final :class:`~torch_geometric.data.Data` object from the
    raw subgraph record (:meth:`_build_data`).  This keeps the base class free
    of assumptions about feature/label naming or how many tensors to fetch.

    Args:
        graph_store (DatabaseGraphStore): Database graph store used to fetch
            induced subgraphs.
        feature_store (DatabaseFeatureStore): Database feature store used to
            fetch node features and labels inside :meth:`_build_data`.
        batch_size (int): Number of seed nodes (or steps) per subgraph.
        num_steps (int): Number of iterations (mini-batches) per epoch.
            (default: :obj:`1`)
        sample_coverage (int): Target average visit count per node for norm
            pre-computation.  ``0`` disables norm computation.
            (default: :obj:`0`)
        save_dir (str, optional): Directory for caching computed norms.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        graph_store: DatabaseGraphStore,
        feature_store: DatabaseFeatureStore,
        batch_size: int,
        num_steps: int,
        sample_coverage: int = 50,
        save_dir: Optional[str] = None,
    ) -> None:
        self.graph_store = graph_store
        self.feature_store = feature_store
        self._batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir

        # Built once; subclasses may override _setup() to do extra work that
        # requires these attrs to already be in place.
        self._subgraph_query = self._build_subgraph_query()
        self._setup()

        super().__init__(
            dataset=self,
            batch_size=1,
            collate_fn=self._collate,
        )

        self.N: int = self._get_total_nodes()
        self.node_norm: Optional[Dict[int, float]] = None
        self._default_norm: float = 1.0

        if self.sample_coverage > 0:
            path = osp.join(save_dir or "", self._filename)
            if save_dir is not None and osp.exists(path):
                self.node_norm, self._default_norm = torch.load(
                    path, weights_only=True)
            else:
                self.node_norm, self._default_norm = self._compute_norm()
                if save_dir is not None:
                    torch.save((self.node_norm, self._default_norm), path)

    def _setup(self) -> None:
        """Post-init hook called after all base attrs are set.

        Override in subclasses to build variant-specific queries or
        data structures.  The base implementation is a no-op.
        """

    @abstractmethod
    def _sample_nodes(self) -> List[int]:
        """Return global node IDs forming one subgraph sample."""

    @abstractmethod
    def _get_total_nodes(self) -> int:
        """Return the total number of nodes in the graph."""

    @abstractmethod
    def _build_subgraph_query(self) -> str:
        """Return a database query that fetches the induced subgraph.

        The query must accept a ``$nodes`` parameter (list of global node
        IDs) and return at minimum:

        * ``nodes``   — ordered list of global node IDs in the subgraph.
        * ``edges`` — list of ``[src_id, dst_id]`` for induced edges.

        Any extra fields (e.g. ``splits``) are passed through to
        :meth:`_build_masks` so subclasses can construct training masks.
        """

    @abstractmethod
    def _build_data(
        self,
        record: dict,
        nodes: List[int],
        edge_index: torch.Tensor,
        node_norm: Optional[torch.Tensor],
    ) -> Data:
        r"""Assemble the final :class:`~torch_geometric.data.Data` object.

        Called by :meth:`_collate` after the edge index and norm weights have
        been decoded.  Feature and label retrieval (e.g. via
        :meth:`~torch_geometric.data.DatabaseFeatureStore.multi_get_tensor`)
        as well as any split-mask construction happen here, keeping the base
        class free of assumptions about attr naming or tensor count.

        Args:
            record (dict): Raw record returned by the subgraph query.  May
                contain extra fields (e.g. ``splits``) beyond ``nodes``
                and ``edges``.
            nodes (List[int]): Global node IDs in subgraph order.
            edge_index (torch.Tensor): Local-index COO edge tensor
                ``[2, E]`` already decoded from ``record["edges"]``.
            node_norm (torch.Tensor, optional): Per-node importance weights
                ``[N]``, or ``None`` when ``sample_coverage == 0``.

        Returns:
            :class:`~torch_geometric.data.Data`: Fully assembled mini-batch.
        """

    @property
    def _filename(self) -> str:
        """Cache filename for serialised norms.  Override in subclasses."""
        return f"database_graphsaint_{self.sample_coverage}.pt"

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Sample one subgraph from the database backend.

        The ``idx`` argument is ignored — every call performs a fresh sample,
        matching PyG's ``GraphSAINTSampler.__getitem__`` behaviour.
        """
        nodes = self._sample_nodes()
        if not nodes:
            return None

        record = self.graph_store._fetch_subgraph(self._subgraph_query,
                                                  {"nodes": nodes})
        return record

    def _collate(self, data_list: List) -> Data:
        """Decode shared subgraph structure, delegate to :meth:`_build_data`.

        Handles the two pieces every variant needs identically — the COO edge
        index (local coordinates) and the optional ``node_norm`` weights —
        then hands everything off to the subclass.
        """
        assert len(data_list) == 1
        record = data_list[0]

        if record is None or not record.get("nodes"):
            return self._build_data(
                record={},
                nodes=[],
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                node_norm=None,
            )

        nodes: List[int] = list(record["nodes"])

        # Edge index in local coordinates.
        edges = record.get("edges") or []
        if edges:
            global_to_local = {nid: i for i, nid in enumerate(nodes)}
            row = torch.tensor([global_to_local[e[0]] for e in edges],
                               dtype=torch.long)
            col = torch.tensor([global_to_local[e[1]] for e in edges],
                               dtype=torch.long)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Importance-sampling weights.
        node_norm: Optional[torch.Tensor] = None
        if self.sample_coverage > 0 and self.node_norm is not None:
            norms = [
                self.node_norm.get(nid, self._default_norm) for nid in nodes
            ]
            node_norm = torch.tensor(norms, dtype=torch.float)

        return self._build_data(record, nodes, edge_index, node_norm)

    def _compute_norm(self) -> Tuple[Dict[int, float], float]:
        """Pre-sample subgraphs to estimate per-node normalisation weights.

        Runs until ``total_sampled_nodes >= N * sample_coverage``.
        Mirrors PyG's formula: ``node_norm[v] = num_samples / count[v] / N``.
        Nodes with zero visits use ``count = 0.1`` (same as PyG).
        """
        node_count: defaultdict = defaultdict(float)
        total_sampled = 0
        num_samples = 0
        target = self.N * self.sample_coverage

        pbar = (tqdm(
            total=int(target),
            desc=f"Computing {self.__class__.__name__} normalization"))

        while total_sampled < target:
            nodes = self._sample_nodes()
            if not nodes:
                continue
            for nid in nodes:
                node_count[nid] += 1
            total_sampled += len(nodes)
            num_samples += 1
            if pbar is not None:
                pbar.update(len(nodes))

        if pbar is not None:
            pbar.close()

        default_norm = (float(num_samples) / 0.1 /
                        self.N if self.N > 0 else 1.0)
        node_norm = {
            nid: float(num_samples) / max(count, 0.1) / self.N
            for nid, count in node_count.items()
        }
        return node_norm, default_norm
