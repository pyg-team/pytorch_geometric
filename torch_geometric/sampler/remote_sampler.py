from abc import abstractmethod
from typing import List, Union

import torch
from torch import Tensor

from torch_geometric.data.graph_store import RemoteGraphStore
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)


class RemoteSampler(BaseSampler):
    r"""Abstract base class for samplers that push multi-hop neighbor sampling
    into a remote graph database via a native query language (e.g. Cypher,
    GSQL).

    The base class handles the full :meth:`sample_from_nodes` pipeline:

    1. Extract seed node IDs from a
       :class:`~torch_geometric.sampler.NodeSamplerInput`.
    2. Build backend query parameters via :meth:`_seed_params`.
    3. Execute the pre-compiled query against the remote store via
       ``RemoteGraphStore._fetch_subgraph``.
    4. Optionally extract per-hop node lists via :meth:`_extract_nodes_by_hop`.
    5. Decode the raw record into COO tensors via
       ``RemoteGraphStore._decode_subgraph``.
    6. Wrap the result in a :class:`~torch_geometric.sampler.SamplerOutput`.

    Subclasses must implement :meth:`_build_node_fanout_query`.  The query is
    compiled once at construction time and reused for every mini-batch.

    Args:
        graph_store (RemoteGraphStore): The remote graph store that executes
            queries and decodes results.
        num_neighbors (List[int]): Number of neighbors to sample per hop.
            Negative values mean "take all".
    """
    def __init__(
        self,
        graph_store: RemoteGraphStore,
        num_neighbors: List[int],
        track_nodes_by_hop: bool = False,
    ):
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors
        # Track nodes by hop needed for pre aggregation of weights.
        self.track_nodes_by_hop = track_nodes_by_hop
        self.last_nodes_by_hop: List[List[int]] = []
        self.node_sampling_query = self._build_node_sampling_query()
        self.edge_sampling_query = self._build_edge_sampling_query()

    def _build_node_sampling_query(self) -> str:
        """Compile a native query for node-seed multi-hop neighbor sampling.

        The query must return ``edge_pairs`` (list of ``[src_id, dst_id]``
        pairs using the backend's global node IDs) and ``nodes_by_hop``
        (list of per-hop node ID lists, seeds first).
        """
        return None

    def _build_edge_sampling_query(self) -> str:
        """Compile a native query for edge-seed multi-hop neighbor sampling.

        Override this method to support :meth:`sample_from_edges`.
        """
        return None

    @abstractmethod
    def _build_query_params(self, seeds: Tensor, **kwargs) -> dict:
        """Return the query parameter dict for the given seed node IDs.

        The default passes seeds under the key ``seed_ids``, which is the
        convention used by Cypher queries in this codebase.  Override if your
        backend expects a different parameter name or type.

        Args:
            seeds (torch.Tensor): 1-D int64 tensor of seed node IDs.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            dict: Parameter dict passed to
                ``RemoteGraphStore._fetch_subgraph``.
        """

    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Sample a subgraph starting from the seed nodes in *index*.

        Args:
            index (NodeSamplerInput): Seed node inputs.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            SamplerOutput: Sampled subgraph as local COO tensors.
        """
        if self.node_sampling_query is None:
            raise ValueError("Node sampling query is not built.")

        seeds = index.node.to(torch.int64)
        seed_time = getattr(index, "time", None)

        params = self._build_query_params(seeds, **kwargs)
        record = self.graph_store._fetch_subgraph(self.node_sampling_query,
                                                  params)
        if self.track_nodes_by_hop:
            self.last_nodes_by_hop = self._extract_nodes_by_hop(record)
        node, row, col = self.graph_store._decode_subgraph(record, seeds)

        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=None,
            batch=None,
            metadata=(seeds, seed_time),
        )

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling=None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Not implemented by default.

        Override :meth:`_build_edge_sampling_query` and this method to support
        edge-seed sampling.
        """
        if self.edge_sampling_query is None:
            raise ValueError("Edge sampling query is not built.")

        seeds = index.edge.to(torch.int64)
        seed_time = getattr(index, "time", None)
        params = self._build_query_params(seeds, **kwargs)
        record = self.graph_store._fetch_subgraph(self.edge_sampling_query,
                                                  params)
        node, row, col = self.graph_store._decode_subgraph(record, seeds)
        return SamplerOutput(node=node, row=row, col=col, edge=None,
                             batch=None, metadata=(seeds, seed_time))
