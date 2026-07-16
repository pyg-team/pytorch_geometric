from typing import Any

import torch
from torch import Tensor

from torch_geometric.data.database_graph_store import DatabaseGraphStore
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)


class DatabaseSampler(BaseSampler):
    r"""Abstract base class for samplers that push multi-hop neighbor sampling
    into a database via a native query language (e.g Cypher, SQL, SQL/PGQ).

    Pipeline for both :meth:`sample_from_nodes` and :meth:`sample_from_edges`:

    1. Extract seed IDs from the sampler input.
    2. Build query parameters via :meth:`_build_node_query_params` or
       :meth:`_build_edge_query_params`.
    3. Execute the pre-compiled query through
       :meth:`~torch_geometric.data.DatabaseGraphStore.query_db`.
    4. Decode the raw record via :meth:`_decode_node_sampling_record` or
       :meth:`_decode_edge_sampling_record`.
    5. Wrap into :class:`~torch_geometric.sampler.SamplerOutput` or
       :class:`~torch_geometric.sampler.HeteroSamplerOutput`.

    Subclasses must implement :meth:`_build_node_sampling_query` and/or
    :meth:`_build_edge_sampling_query`, plus the matching params builders
    and decoders. Queries are compiled once at construction and reused.

    Args:
        graph_store (DatabaseGraphStore): Graph store that executes queries.
        is_hetero (bool): Whether the graph is heterogeneous.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        graph_store: DatabaseGraphStore,
        is_hetero: bool = False,
    ):
        self.graph_store = graph_store
        self._is_hetero = is_hetero
        self.node_sampling_query = self._build_node_sampling_query()
        self.edge_sampling_query = self._build_edge_sampling_query()

    @property
    def is_hetero(self) -> bool:
        r"""Whether the sampler operates on a heterogeneous graph."""
        return self._is_hetero

    def _build_node_sampling_query(self) -> str | None:
        r"""Compile native query for node-seed multi-hop sampling.

        Override to enable :meth:`sample_from_nodes`. Returns :obj:`None`
        if node-seed sampling is not supported.
        """
        return None

    def _build_edge_sampling_query(self) -> str | None:
        r"""Compile native query for edge-seed multi-hop sampling.

        Override to enable :meth:`sample_from_edges`. Returns :obj:`None`
        if edge-seed sampling is not supported.
        """
        return None

    def _build_node_query_params(self, seeds: Tensor, **kwargs) -> dict | None:
        r"""Build query parameter dict for seed node IDs.

        Args:
            seeds (torch.Tensor): 1-D int64 tensor of seed node IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Parameters passed to
                :meth:`~torch_geometric.data.DatabaseGraphStore.query_db`.
        """
        return None

    def _build_edge_query_params(self, seeds: Tensor, **kwargs) -> dict | None:
        r"""Build query parameter dict for seed edge endpoint IDs.

        Same contract as :meth:`_build_node_query_params`; ``seeds`` are the
        unique endpoint node IDs of the seed edges.
        """
        return None

    @staticmethod
    def _empty_result(
        seed_nodes: Tensor, is_hetero: bool
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[dict, dict, dict]:
        r"""Return empty ``(node, row, col)`` triple for the given layout."""
        if is_hetero:
            return {}, {}, {}
        empty = torch.zeros(0, dtype=torch.long)
        return seed_nodes, empty, empty

    def _decode_node_sampling_record(self, record: Any,
                                     seeds: Tensor) -> Any | None:
        r"""Decode a raw record into ``(node, row, col)`` for node sampling.

        Args:
            record: Raw record returned by
                :meth:`~torch_geometric.data.DatabaseGraphStore.query_db`.
            seeds (torch.Tensor): Seed node IDs used to build the query.

        Returns:
            Tuple of tensors (homogeneous) or dicts (heterogeneous), or
            :obj:`None` if the record is empty.
        """
        raise NotImplementedError

    def _decode_edge_sampling_record(self, record: Any,
                                     seeds: Tensor) -> Any | None:
        r"""Decode a raw record into ``(node, row, col)`` for edge sampling.

        Same contract as :meth:`_decode_node_sampling_record`.
        """
        raise NotImplementedError

    def _build_output(
        self,
        node,
        row,
        col,
        seeds: Tensor,
        seed_time,
        input_type: str | None = None,
    ) -> SamplerOutput | HeteroSamplerOutput:
        r"""Wrap ``(node, row, col)`` into a homo or hetero sampler output."""
        if self._is_hetero:
            if input_type is not None and input_type not in node:
                node = {**node, input_type: seeds}
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
    ) -> SamplerOutput | HeteroSamplerOutput:
        r"""Sample a subgraph starting from the seed nodes in ``index``.

        Args:
            index (NodeSamplerInput): Seed node inputs.
            **kwargs: Forwarded to :meth:`_build_node_query_params`.

        Returns:
            Sampled subgraph as local COO tensors.
        """
        if not self.node_sampling_query:
            raise ValueError("Node sampling query is not built.")

        seeds = index.node.to(torch.int64)
        seed_time = getattr(index, "time", None)

        params = self._build_node_query_params(seeds, **kwargs)
        if not params:
            raise ValueError(
                "Query parameters are empty. Check the implementation of "
                "_build_node_query_params.")
        record = self.graph_store.query_db(self.node_sampling_query, params)
        decoded = self._decode_node_sampling_record(record, seeds)
        if decoded is None:
            node, row, col = self._empty_result(seeds, self._is_hetero)
        else:
            node, row, col = decoded

        input_type = getattr(index, "input_type", None)
        return self._build_output(node, row, col, seeds, seed_time, input_type)

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling=None,
        **kwargs,
    ) -> SamplerOutput | HeteroSamplerOutput:
        r"""Sample a subgraph starting from the seed edges in ``index``.

        Args:
            index (EdgeSamplerInput): Seed edge inputs.
            neg_sampling (NegativeSampling, optional): Negative sampling
                configuration. Not supported; must be :obj:`None`.
                (default: :obj:`None`)
            **kwargs: Forwarded to :meth:`_build_edge_query_params`.

        Returns:
            Sampled subgraph as local COO tensors.
        """
        if neg_sampling is not None:
            raise NotImplementedError(
                "negative sampling is not supported by DatabaseSampler; "
                f"got neg_sampling={neg_sampling!r}.")

        if not self.edge_sampling_query:
            raise ValueError("Edge sampling query is not built.")

        row = index.row.to(torch.int64)
        col = index.col.to(torch.int64)
        seeds = torch.cat([row, col]).unique()
        seed_time = getattr(index, "time", None)

        params = self._build_edge_query_params(seeds, **kwargs)
        if not params:
            raise ValueError(
                "Query parameters are empty. Check the implementation of "
                "_build_edge_query_params.")
        record = self.graph_store.query_db(self.edge_sampling_query, params)
        decoded = self._decode_edge_sampling_record(record, seeds)
        if decoded is None:
            node, row, col = self._empty_result(seeds, self._is_hetero)
        else:
            node, row, col = decoded

        input_type = getattr(index, "input_type", None)
        return self._build_output(node, row, col, seeds, seed_time, input_type)
