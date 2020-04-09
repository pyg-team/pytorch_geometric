from itertools import combinations
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Union

import torch
from torch_scatter import scatter, segment_csr
from torch_sparse import SparseTensor

from .inspector import Inspector
from .collector import (
    Collector,
    SparseAdjFusedCollector,
    DenseAdjFusedCollector,
    EdgeIndexSparseCollector,
    SparseAdjSparseCollector,
    SparseAdjPartialCollector,
    DenseAdjPartialCollector,
)
from .helpers import abs_dim, unsqueeze


class MessagePassing(torch.nn.Module):

    AdjType = Union[torch.Tensor, SparseTensor]
    adj_formats: List[str] = ['edge_index', 'sparse', 'dense']
    mp_formats: List[str] = ['fused', 'sparse', 'dense']

    def __init__(self, aggr: str = "add", flow: str = "source_to_target",
                 mp_format: Optional[str] = None, node_dim: int = -2,
                 partial_fill_value: float = 0.,
                 partial_max_deg: Optional[int] = None,
                 partial_binning: bool = True, torchscript: bool = False):
        super(MessagePassing, self).__init__()

        self.aggr: str = aggr
        self.flow: str = flow
        self.mp_format: Optional[str] = mp_format
        self.node_dim: int = node_dim
        self.torchscript: bool = torchscript  # TODO: This has no effect yet.

        assert self.aggr in ['add', 'sum', 'mean', 'max', None]
        assert self.flow in ['source_to_target', 'target_to_source']
        assert self.mp_format in self.mp_formats + [None]

        self.inspector = Inspector(self)
        self.inspector.inspect(self.sparse_message_and_aggregate)
        self.inspector.inspect(self.dense_message_and_aggregate)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.partial_aggregate, pop_first=True)

        # In case `aggregate` or `partial_aggregate` is implemented by the
        # user, we do not want to make use of the predefined aggregations.
        if (self.inspector.implements('aggregate')
                or self.inspector.implements('partial_aggregate')):
            self.aggr = None

        # In case of partial "max" aggregation, it is faster to already fill
        # invalid entries with `-inf` instead of `0`.
        if self.aggr is not None:
            self.partial_fill_value = 0 if aggr != 'max' else float('-inf')

        # An `OrderedDict` that dictates the preference of message passing.
        coll = OrderedDict()
        if self.supports_sparse_fused_format():
            coll[('sparse_adj', 'fused')] = SparseAdjFusedCollector(self)
        if self.supports_dense_fused_format():
            coll[('dense_adj', 'fused')] = DenseAdjFusedCollector(self)
        if self.supports_sparse_format():
            coll[('edge_index', 'sparse')] = EdgeIndexSparseCollector(self)
            coll[('sparse_adj', 'sparse')] = SparseAdjSparseCollector(self)
        if self.supports_partial_format():
            # coll[('sparse_adj', 'partial')] = SparseAdjPartialCollector(
            #     self, partial_fill_value, partial_max_deg, partial_binning)
            coll[('dense_adj', 'partial')] = DenseAdjPartialCollector(
                self, partial_fill_value)
        self.collectors = coll

        if self.inspector.implements('update'):
            raise TypeError(
                (f'Updating node embeddings via '
                 f'`{self.__class__.__name__}.update` inside message '
                 f'propagation is no longer supported and should be performed '
                 f'on its own after `{self.__class__.__name__}.propagate`.'))

        # We cache the determined `mp_format` for faster re-access.
        self.__cached_mp_format__: Dict[str, str] = {}

        # Support for `GNNExplainer`.
        self.__explain__: bool = False
        self.__edge_mask__: bool = None

    def supports_sparse_fused_format(self) -> bool:
        return self.inspector.implements('sparse_message_and_aggregate')

    def supports_dense_fused_format(self) -> bool:
        return self.inspector.implements('dense_message_and_aggregate')

    def supports_sparse_format(self) -> bool:
        return (self.inspector.implements('message')
                and (self.inspector.implements('aggregate')
                     or self.aggr is not None))

    def supports_partial_format(self) -> bool:
        return (self.inspector.implements('message')
                and (self.inspector.implements('partial_aggregate')
                     or self.aggr is not None))

    def get_adj_format(self, adj_type: AdjType) -> str:
        # "edge_index": torch.LongTensor of shape [2, *].
        if (torch.is_tensor(adj_type) and adj_type.dim() == 2
                and adj_type.size(0) == 2 and adj_type.dtype == torch.long):
            return 'edge_index'

        # "sparse_adj": torch_sparse.SparseTensor.
        elif isinstance(adj_type, SparseTensor):
            return 'sparse_adj'

        # "dense_adj": *Any* other torch.Tensor.
        elif torch.is_tensor(adj_type):
            return 'dense_adj'

        raise ValueError(
            (f'Encountered an invalid object for `adj_type` in '
             f'`{self.__class__.__name__}.propagate`. Supported types are (1) '
             f'sparse edge indices of type `torch.LongTensor` with shape '
             f'`[2, num_edges]`, (2) sparse adjacency matrices of type '
             f'`torch_sparse.SparseTensor`, or (3) dense adjacency matrices '
             f'of type `torch.Tensor`.'))

    def get_mp_format(self, adj_format: str) -> str:
        # Set to user-desired format (if present).
        if self.mp_format is not None:
            return self.mp_format

        # Use already determined cached message passing format (if present).
        elif adj_format in self.__cached_mp_format__:
            return self.__cached_mp_format__[adj_format]

        # Detect message passing format by iterating over remaining collectors.
        elif adj_format not in self.__cached_mp_format__:
            for key in self.collectors.keys():
                if key[0] == adj_format:
                    self.__cached_mp_format__[adj_format] = key[1]
                    return key[1]

        raise TypeError(
            (f'Could not detect a valid message passing implementation '
             f'for adjacency format "{adj_format}".'))

    def __get_collector__(self, adj_format: str, mp_format: str) -> Collector:
        collector = self.collectors.get((adj_format, mp_format), None)

        if collector is None:
            raise TypeError(
                (f'Could not detect a valid message passing implementation '
                 f'for adjacency format "{adj_format}" and message passing '
                 f'format "{mp_format}".'))

        return collector

    def propagate(self, adj_type: AdjType, size: Optional[Tuple[int]] = None,
                  **kwargs) -> torch.Tensor:

        adj_format = self.get_adj_format(adj_type)
        mp_format = self.get_mp_format(adj_format)

        # For `GNNExplainer`, we require "sparse" message passing based on
        # "edge_index" adjacency format since this allows us to easily inject
        # the `edge_mask` into the message passing computation.
        # NOTE: Technically, it is still possible to provide this for all
        # types of message passing. However, for other formats it is a lot
        # hackier to implement, so we leave this for future work at the moment.
        if self.__explain__:
            if (adj_format == 'sparse_adj' or adj_format == 'dense_adj'
                    or not self.supports_sparse_format()):
                raise TypeError(
                    (f'`{self.__class__.__name__}.propagate` only supports '
                     f'`GNNExplainer` capabilties for "sparse" message '
                     f'passing based on "edge_index" adjacency format.`'))
            mp_format = 'sparse'

        # Customized flow direction is deprecated for "new" adjacency matrix
        # formats, i.e., "sparse_adj" and "dense_adj".
        if ((adj_format == 'sparse_adj' or adj_format == 'dense_adj')
                and self.flow == 'target_to_source'):
            raise TypeError(
                (f'Flow direction "target_to_source" is invalid for message '
                 f'passing based on adjacency matrices. If you really want to '
                 f'make use of reverse message passing flow, pass in the '
                 f'transposed adjacency matrix to '
                 f'`{self.__class__.__name__}.propagate`, e.g., via '
                 f'`adj_t.t()`.'))

        collector = self.__get_collector__(adj_format, mp_format)

        # Perform "conditional" message passing.
        if (adj_format, mp_format) == ('sparse_adj', 'fused'):
            kwargs = collector.collect(adj_type, size, kwargs)
            inputs = self.inspector.distribute(
                self.sparse_message_and_aggregate, kwargs)
            return self.sparse_message_and_aggregate(**inputs)

        elif (adj_format, mp_format) == ('dense_adj', 'fused'):
            kwargs = collector.collect(adj_type, size, kwargs)
            inputs = self.inspector.distribute(
                self.dense_message_and_aggregate, kwargs)
            return self.dense_message_and_aggregate(**inputs)

        elif mp_format == 'sparse':
            kwargs = collector.collect(adj_type, size, kwargs)
            inputs = self.inspector.distribute(self.message, kwargs)
            out = self.message(**inputs)

            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # If the edge sizes do not match, we assume that the message
                # passing implementation has added self-loops to the graph
                # before calling `self.propagate`. This is just educated guess,
                # but should be sufficient in most cases.
                if out.size(0) != edge_mask.size(0):
                    # NOTE: This currently does only work for "edge_index"
                    # format, but could be enhanced to also support
                    # "sparse_adj" format.
                    # TODO: Make use of unified `add_self_loops` interface to
                    # implement this.
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * unsqueeze(edge_mask, dim=-1, length=out.dim() - 1)

            inputs = self.inspector.distribute(self.aggregate, kwargs)
            return self.aggregate(out, **inputs)

        elif (adj_format, mp_format) == ('dense_adj', 'partial'):
            message_kwargs = collector.collect_message(adj_type, kwargs)
            inputs = self.inspector.distribute(self.message, message_kwargs)
            out = self.message(**inputs)
            out, kwargs = collector.collect_aggregate(adj_type, out, kwargs,
                                                      message_kwargs)
            inputs = self.inspector.distribute(self.partial_aggregate, kwargs)
            return self.partial_aggregate(out, **inputs)

    def sparse_message_and_aggregate(self) -> torch.Tensor:
        raise NotImplementedError

    def dense_message_and_aggregate(self) -> torch.Tensor:
        raise NotImplementedError

    def message(self) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:

        if ptr is not None:
            ptr = unsqueeze(ptr, dim=0, length=abs_dim(inputs, self.node_dim))
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def partial_aggregate(self, inputs, edge_mask) -> torch.Tensor:

        dim = self.node_dim + 1 if self.node_dim > 0 else self.node_dim

        if self.aggr in ['sum', 'add']:
            return inputs.sum(dim=dim)

        elif self.aggr == 'mean':
            out = inputs.sum(dim=dim)
            deg = edge_mask.sum(dim=-1)
            deg = unsqueeze(deg, dim=-1, length=out.dim() - deg.dim())
            out = out / deg
            out.masked_fill_((out == float('inf')) | (out == float('-inf')), 0)
            return out

        elif self.aggr == 'max':
            out = inputs.max(dim=dim)[0]
            out.masked_fill_(out == float('-inf'), 0)
            return out

    @torch.no_grad()
    def check_consistency(self, sparse_adj_t: SparseTensor,
                          static_graph: bool = False, rtol: float = 1e-05,
                          atol: float = 1e-08, **kwargs) -> bool:

        keys = list(self.collectors.keys())

        if len(keys) < 2:
            return True

        outs = []
        old_mp_format = self.mp_format
        for (adj_format, mp_format) in keys:
            self.mp_format = mp_format  # Force message passing format.

            if adj_format == 'edge_index':
                row, col, edge_attr = sparse_adj_t.t().coo()  # Transpose.
                edge_index = torch.stack([row, col], dim=0)
                size = list(sparse_adj_t.sparse_sizes())[::-1]
                out = self.propagate(edge_index, size=size,
                                     edge_attr=edge_attr, **kwargs)

            elif adj_format == 'sparse_adj':
                out = self.propagate(sparse_adj_t, **kwargs)

            elif adj_format == 'dense_adj':
                dense_adj_t = sparse_adj_t.to_dense()
                if static_graph:
                    dense_adj_t = dense_adj_t.unsqueeze(0)
                out = self.propagate(dense_adj_t, **kwargs)

            outs.append((adj_format, mp_format, out))
        self.mp_format = old_mp_format

        # Perform `allclose` checks between all combinations.
        for out1, out2 in combinations(outs, 2):
            if not torch.allclose(out1[2], out2[2], rtol=rtol, atol=atol):
                return False

        return True

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
