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
    2. Build backend query parameters via :meth:`_build_query_params`.
    3. Execute the pre-compiled query against the remote store via
       ``RemoteGraphStore.sample_subgraph``.
    4. Wrap the result in a :class:`~torch_geometric.sampler.SamplerOutput`.

    Subclasses must implement :meth:`_build_node_sampling_query`
    or :meth:`_build_edge_sampling_query` to support node-
    or edge-seed sampling. The query is compiled once at
    construction time and reused for every mini-batch.

    Args:
        graph_store (RemoteGraphStore): The remote graph store that executes
            queries and returns results.
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
        """Build the query parameter dict for the given seed node IDs.

        Args:
            seeds (torch.Tensor): 1-D int64 tensor of seed node IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Parameter dict passed to
                ``RemoteGraphStore.sample_subgraph``.
        """

    def _build_output(
        self,
        node,
        row,
        col,
        seeds: Tensor,
        seed_time,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Wrap ``(node, row, col)`` into the correct output type.

        Returns :class:`~torch_geometric.sampler.HeteroSamplerOutput` when
        *node* is a dict (heterogeneous result from the graph store), and
        :class:`~torch_geometric.sampler.SamplerOutput` otherwise.
        """
        if isinstance(node, dict):
            return HeteroSamplerOutput(
                node=node,
                row=row,
                col=col,
                edge={et: None
                      for et in row},
                metadata=(seeds, seed_time),
            )
        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=None,
            batch=None,
            metadata=(seeds, seed_time),
        )

    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Sample a subgraph starting from the seed nodes in *index*.

        Args:
            index (NodeSamplerInput): Seed node inputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[SamplerOutput, HeteroSamplerOutput]: Sampled subgraph as
            local COO tensors.  Returns
            :class:`~torch_geometric.sampler.HeteroSamplerOutput` when the
            graph store's :meth:`_decode_subgraph` returns dicts (i.e. the
            graph is heterogeneous).
        """
        if not self.node_sampling_query:
            raise ValueError("Node sampling query is not built.")

        seeds = index.node.to(torch.int64)
        seed_time = getattr(index, "time", None)

        params = self._build_query_params(seeds, **kwargs)
        node, row, col = self.graph_store.sample_subgraph(
            self.node_sampling_query, params, seeds)

        return self._build_output(node, row, col, seeds, seed_time)

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling=None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Sample a subgraph starting from the seed edges in *index*.

        Args:
            index (EdgeSamplerInput): Seed edge inputs.
            neg_sampling (NegativeSampling, optional): The negative sampling
                configuration. (default: :obj:`None`)
            **kwargs: Additional keyword arguments.

        Returns:
            Union[SamplerOutput, HeteroSamplerOutput]: Sampled subgraph as
            local COO tensors.
        """
        if not self.edge_sampling_query:
            raise ValueError("Edge sampling query is not built.")

        seeds = index.edge.to(torch.int64)
        seed_time = getattr(index, "time", None)
        params = self._build_query_params(seeds, **kwargs)
        node, row, col = self.graph_store.sample_subgraph(
            self.edge_sampling_query, params, seeds)

        return self._build_output(node, row, col, seeds, seed_time)
