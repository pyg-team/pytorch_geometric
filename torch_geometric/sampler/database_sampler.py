from abc import abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.data.graph_store import (
    DatabaseGraphStore,
    HeterogeneousSchema,
    HomogeneousSchema,
    ResultSchema,
)
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)


class DatabaseSampler(BaseSampler):
    r"""Abstract base class for samplers that push multi-hop neighbor sampling
    into a database via a native query language (e.g. Cypher, GSQL).

    The base class handles the full :meth:`sample_from_nodes` pipeline:

    1. Extract seed node IDs from a
       :class:`~torch_geometric.sampler.NodeSamplerInput`.
    2. Build database query parameters via :meth:`_build_query_params`.
    3. Execute the pre-compiled query against the database store via
       ``DatabaseGraphStore.sample_subgraph``.
    4. Wrap the result in a :class:`~torch_geometric.sampler.SamplerOutput`.

    Subclasses must implement :meth:`_build_node_sampling_query`
    or :meth:`_build_edge_sampling_query` to support node-
    or edge-seed sampling. The query is compiled once at
    construction time and reused for every mini-batch.

    Args:
        graph_store (DatabaseGraphStore): The database graph store that
            executes queries and returns results.
        num_neighbors (List[int]): Number of neighbors to sample per hop.
            Negative values mean "take all".
        schema (ResultSchema, optional): Description of the record shape
            returned by the graph store, used by the store's decoder.
            Defaults to :class:`HomogeneousSchema`.
    """
    def __init__(
        self,
        graph_store: DatabaseGraphStore,
        schema: Optional[ResultSchema] = None,
        is_hetero: bool = False,
    ):
        self.graph_store = graph_store
        self.node_sampling_query = self._build_node_sampling_query()
        self.edge_sampling_query = self._build_edge_sampling_query()

        if schema is None:
            schema = self._build_schema(is_hetero)
        if schema.is_hetero != is_hetero:
            raise ValueError(
                f"schema.is_hetero={schema.is_hetero} contradicts "
                f"is_hetero={is_hetero}; pass only one or make them agree.")
        self.schema = schema

    def _build_schema(self, is_hetero: bool) -> ResultSchema:
        r"""Default schema picked when the caller does not pass one
        explicitly.

        Override in a subclass to return a custom :class:`ResultSchema`.
        The returned schema's :attr:`is_hetero` must match the *is_hetero*
        constructor argument; mismatch raises :class:`ValueError`.
        """
        return HeterogeneousSchema() if is_hetero else HomogeneousSchema()

    @property
    def is_hetero(self) -> bool:
        r"""Whether the active schema is heterogeneous.  Always consistent
        with :attr:`self.schema`.
        """
        return self.schema.is_hetero

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
                ``DatabaseGraphStore.sample_subgraph``.
        """

    def _build_output(
        self,
        node,
        row,
        col,
        seeds: Tensor,
        seed_time,
        input_type: Optional[str] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Wrap ``(node, row, col)`` into the correct output type.

        Dispatches on :attr:`self.schema.is_hetero`.  For heterogeneous
        schemas, splices *seeds* into ``node[input_type]`` if missing — this
        rescues the empty-result case where the decoder produced no nodes
        for the seed's type.
        """
        if self.schema.is_hetero:
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
            self.node_sampling_query, params, seeds, self.schema)

        input_type = getattr(index, "input_type", None)
        return self._build_output(node, row, col, seeds, seed_time, input_type)

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
            self.edge_sampling_query, params, seeds, self.schema)

        input_type = getattr(index, "input_type", None)
        return self._build_output(node, row, col, seeds, seed_time, input_type)
