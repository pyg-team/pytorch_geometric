from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_scatter import scatter, segment_csr
from torch_sparse import SparseTensor

from .inspector import Inspector
from .collector import (
    Collector,
    EdgeIndexSparseCollector,
    SparseAdjSparseCollector,
    SparseAdjFusedCollector,
    DenseAdjFusedCollector,
    DenseAdjPartialCollector,
)


class MessagePassing(torch.nn.Module):

    AdjType = Union[torch.Tensor, SparseTensor]
    adj_formats: List[str] = ['edge_index', 'sparse', 'dense']
    mp_formats: List[str] = ['fused', 'sparse', 'dense']

    __collectors__: Dict[str, Collector] = {
        ('edge_index', 'sparse'): EdgeIndexSparseCollector(),
        ('sparse_adj', 'fused'): SparseAdjFusedCollector(),
        ('sparse_adj', 'sparse'): SparseAdjSparseCollector(),
        ('dense_adj', 'fused'): DenseAdjFusedCollector(),
        ('dense_adj', 'partial'): DenseAdjPartialCollector(),
    }

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
        self.partial_fill_value: float = partial_fill_value
        self.partial_max_deg: Optional[int] = partial_max_deg
        self.partial_binning: bool = partial_binning
        self.torchscript: bool = torchscript

        assert self.aggr in ['add', 'sum', 'mean', 'max', None]
        assert self.flow in ['source_to_target', 'target_to_source']
        assert self.mp_format in self.mp_formats + [None]

        self.inspector = Inspector(self)
        self.inspector.inspect(self.sparse_message_and_aggregate)
        self.inspector.inspect(self.dense_message_and_aggregate)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.partial_aggregate, pop_first=True)

        # In case of partial "max" aggregation, it is faster to already fill
        # invalid entries with `-inf` instead of `0`.
        if not self.inspector.implements('partial_aggregate'):
            self.partial_fill_value = 0 if aggr != 'max' else float('-inf')

        if self.inspector.implements('update'):
            raise TypeError(
                ('Updating node embeddings via `self.update` inside message '
                 'propagation is no longer supported and should be performed '
                 'on its own after `self.propagate`.'))

        self.__cached_mp_format__ = {}

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
        adj_format = None

        # edge_index: torch.LongTensor of shape [2, *].
        if (torch.is_tensor(adj_type) and adj_type.dim() == 2
                and adj_type.size(0) == 2 and adj_type.dtype == torch.long):
            adj_format = 'edge_index'

        # sparse_adj: torch_sparse.SparseTensor.
        elif isinstance(adj_type, SparseTensor):
            adj_format = 'sparse_adj'

        # dense_adj: *Any* torch.Tensor.
        elif torch.is_tensor(adj_type):
            adj_format = 'dense_adj'

        if adj_format is None:
            raise ValueError(
                ('Encountered an invalid object for `adj_type` in '
                 '`MessagePassing.propagate`. Supported types are (1) sparse '
                 'edge indices of type `torch.LongTensor` with shape '
                 '`[2, num_edges]`, (2) sparse adjacency matrices of type '
                 '`torch_sparse.SparseTensor`, or (3) dense adjacency '
                 'matrices of type `torch.Tensor`.'))

        return adj_format

    def get_mp_format(self, adj_format: str) -> str:
        mp_format = None

        # Use already determined cached message passing format (if present).
        if adj_format in self.__cached_mp_format__:
            mp_format = self.__cached_mp_format__[adj_format]

        # `edge_index` only support "tradional" message passing, i.e. "sparse".
        elif adj_format == 'edge_index':
            mp_format = 'sparse'

        # Set to user-desired format (if present).
        elif self.mp_format is not None:
            mp_format = self.mp_format

        # Always choose `fused` if applicable.
        elif (adj_format == 'sparse_adj'
              and self.supports_sparse_fused_format()):
            mp_format = 'fused'

        elif adj_format == 'dense_adj' and self.supports_dense_fused_format():
            mp_format = 'fused'

        # We prefer "sparse" format over the "partial" format for sparse
        # adjacency matrices since it is faster in general. We therefore only
        # default to "partial" mode if the user wants to implement some fancy
        # customized aggregation scheme.
        elif adj_format == 'sparse_adj' and self.supports_sparse_format():
            mp_format = 'sparse'
        elif adj_format == 'sparse_adj' and self.supports_partial_format():
            mp_format = 'partial'

        # For "dense" adjacencies, we *require* "partial" aggregation.
        elif adj_format == 'dense_adj' and self.supports_partial_format():
            mp_format = 'partial'

        if mp_format is None:
            raise TypeError(
                (f'Could not detect a valid message passing implementation '
                 f'for adjacency format "{adj_format}".'))

        # Fill (or update) the cache.
        self.__cached_mp_format__[adj_format] = mp_format

        return mp_format

    def __get_collector__(self, adj_format: str, mp_format: str) -> Collector:
        collector = self.__collectors__.get((adj_format, mp_format), None)

        if collector is None:
            raise TypeError(
                (f'Could not detect a valid message passing implementation '
                 f'for adjacency format "{adj_format}" and message passing '
                 f'format "{mp_format}".'))

        collector.bind(self)

        return collector

    def propagate(self, adj_type: AdjType, size: Optional[Tuple[int]] = None,
                  **kwargs) -> torch.Tensor:

        adj_format = self.get_adj_format(adj_type)
        mp_format = self.get_mp_format(adj_format)

        # For `GNNExplainer`, we require "sparse" aggregation based on
        # "edge_index" since this allows us to easily inject the `edge_mask`
        # into the message passing computation.
        # NOTE: Technically, it is still possible to provide this for all
        # types of message passing. However, for other formats it is a lot
        # hackier to implement, so we leave this for future work at the moment.
        if self.__explain__:
            if (adj_format == 'sparse_adj' or adj_format == 'dense_adj'
                    or not self.supports_sparse_format()):
                raise TypeError(
                    ('`MessagePassing.propagate` only supports `GNNExplainer` '
                     'capabilties for "sparse" aggregations based on '
                     '"edge_index".`'))
            mp_format = 'sparse'

        # Customized flow direction is deprecated for "new" adjacency matrix
        # formats, i.e., "sparse_adj" and "dense_adj".
        if ((adj_format == 'sparse_adj' or adj_format == 'dense_adj')
                and self.flow == 'target_to_source'):
            raise TypeError(
                ('Flow direction "target_to_source" is invalid for message '
                 'passing based on adjacency matrices. If you really want to '
                 'make use of reverse message passing flow, pass in the '
                 'transposed adjacency matrix to the message passing module, '
                 'e.g., `adj_t.t()`.'))

        # We collect all arguments used for message passing dependening on the
        # determined collector.
        collector = self.__get_collector__(adj_format, mp_format)
        kwargs = collector.collect(adj_type, size, kwargs)

        # Perform "conditional" message passing.
        if adj_format == 'sparse_adj' and mp_format == 'fused':
            inputs = self.inspector.distribute(
                self.sparse_message_and_aggregate, kwargs)
            out = self.sparse_message_and_aggregate(**inputs)

        elif adj_format == 'dense_adj' and mp_format == 'fused':
            inputs = self.inspector.distribute(
                self.dense_message_and_aggregate, kwargs)
            out = self.dense_message_and_aggregate(**inputs)

        elif mp_format == 'sparse':
            inputs = self.inspector.distribute(self.message, kwargs)
            out = self.message(**inputs)

            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # If the edge sizes do not match, we assume that the message
                # passing implementation has added self-loops to the graph
                # before calling `self.propagate`. This is not ideal, but
                # sufficient in most cases.
                if out.size(0) != edge_mask.size(0):
                    # NOTE: This does only work for "edge_index" format, but
                    # could be enhanced to also support "sparse_adj" format.
                    # TODO: Make use of unified `add_self_loops` interface to
                    # implement this.
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * edge_mask.view(-1, 1)

            inputs = self.inspector.distribute(self.aggregate, kwargs)
            out = self.aggregate(out, **inputs)

        elif mp_format == 'partial':
            out = self.__partial_message__(adj_format, kwargs)
            out = self.__partial_aggregate__(adj_format, out, kwargs)

        return out

    def sparse_message_and_aggregate(self) -> torch.Tensor:
        raise NotImplementedError

    def dense_message_and_aggregate(self) -> torch.Tensor:
        raise NotImplementedError

    def message(self) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:

        if self.aggr is None:
            raise NotImplementedError

        if ptr is not None:
            node_dim = self.node_dim
            iters = inputs.dim() + node_dim if node_dim < 0 else node_dim
            for _ in range(iters):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def partial_aggregate(self, inputs, inv_edge_mask) -> torch.Tensor:

        if self.aggr is None:
            raise NotImplementedError

        # TODO: `node_dim` may not be the right choice here.

        if self.aggr in ['sum', 'add']:
            return inputs.sum(self.node_dim)

        elif self.aggr == 'mean':
            out = inputs.sum(self.node_dim)
            deg = inv_edge_mask.size(-1) - inv_edge_mask.sum(dim=-1)
            for _ in range(deg.dim(), out.dim()):
                deg = deg.unsqueeze(-1)
            out = out / deg
            out.masked_fill_((out == float('inf')) | (out == float('-inf')), 0)
            return out

        elif self.aggr == 'max':
            out = inputs.max(dim=self.node_dim)[0]
            out.masked_fill_(out == float('-inf'), 0)
            return out

    def __partial_message__(self, adj_format, kwargs):
        # There is no need to perform "expensive" degree iterations for
        # "partial" message passing based on dense adjacency matrices.
        if adj_format == 'dense_adj':
            kwargs = self.inspector.distribute(self.message, kwargs)
            return self.message(**kwargs)

        else:
            kwargs = self.inspector.distribute(self.message, kwargs)
            keys = list(kwargs.keys())
            num_bins = len(kwargs[keys[0]])

            outs = []
            for i in range(num_bins):
                tmp_kwargs = {key: item[i] for key, item in kwargs.items()}
                outs.append(self.message(**tmp_kwargs))

            return outs

    def __partial_aggregate__(self, adj_format, inputs, kwargs):
        # The "partial" aggregation based on dense adjacency matrices needs
        # special treatment here since we cannot infer the dimensions of
        # `adj_t`, i.e., `edge_attr` without looking at `inputs` for computing
        # `inv_edge_mask`.
        if adj_format == 'dense_adj':
            inputs = inputs.contiguous()

            node_dim = self.node_dim
            node_dim = inputs.dim() + node_dim if node_dim < 0 else node_dim

            # Find all entries zero entries (even for multi-dimensional edge
            # features).
            adj_t = kwargs['edge_attr']
            if adj_t.dim() > node_dim + 2:
                dims = list(range(node_dim + 2, adj_t.dim()))
                inv_edge_mask = adj_t.abs().sum(dims) == 0
            else:
                inv_edge_mask = adj_t == 0
            kwargs['inv_edge_mask'] = inv_edge_mask

            for _ in range(node_dim + 1, inputs.dim()):
                inv_edge_mask = inv_edge_mask.unsqueeze(-1)

            if self.partial_fill_value is not None:
                inputs.masked_fill_(inv_edge_mask, self.partial_fill_value)

            kwargs = self.inspector.distribute(self.partial_aggregate, kwargs)
            return self.partial_aggregate(inputs, **kwargs)

        else:
            kwargs = self.inspector.distribute(self.partial_aggregate, kwargs)
            num_bins = len(inputs)

            outs = []
            for i in range(num_bins):
                tmp_kwargs = {key: item[i] for key, item in kwargs.items()}
                outs.append(self.partial_aggregate(inputs[i], **tmp_kwargs))

            return outs

    def check_propagate_consistency(self, adj_t: SparseTensor,
                                    **kwargs) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
