import inspect
from itertools import chain
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from collections import OrderedDict

import torch
from torch_sparse import SparseTensor

# TODO:
# Check which implementation is implemented
# Debug method that checks if computation is the same

# TODO: edge_perm in padded_index

# Base Class that handles the lifting logic
# -> Gets the arguments and collects them into a dictionary
# -> User Logic: fill argument list
# -> Calls the functions by distributing the required data to the functions

empty = inspect.Parameter.empty
AdjType = Union[torch.Tensor, SparseTensor]

special_args = set([
    'adj_t', 'edge_index_i', 'edge_index_j', 'size_i', 'size_j', 'ptr',
    'index', 'dim_size', 'edge_mask'
])


class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class: Any = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    def inspect(self, func: Callable,
                pop_first: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict({k: v.default for k, v in params.items()})
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names: Optional[List[str]] = None) -> Set[str]:
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys) - special_args

    def distribute(self, func: Callable, kwargs: Dict[str, Any]):
        out = {}
        for key, default in self.params[func.__name__].items():
            data = kwargs.get(key, empty)
            if data is empty:
                if default is empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = default
            out[key] = data
        return out

    def implements(self, func_name: str) -> bool:
        return func_name in self.base_class.__class__.__dict__.keys()


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr: str = "add", flow: str = "source_to_target",
                 format=None, node_dim: int = 0, partial_max_deg: int = -1,
                 partial_binning: bool = True, torchscript: bool = False):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        self.flow = flow
        self.format = format
        self.node_dim = node_dim
        self.partial_max_deg = partial_max_deg
        self.partial_binning = partial_binning
        self.torchscript = torchscript

        assert self.aggr in ['add', 'sum', 'mean', 'max', None]
        assert self.flow in ['source_to_target', 'target_to_source']
        assert self.format in ['fused', 'sparse', 'partial', None]
        assert self.node_dim >= 0

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.partial_message)
        self.inspector.inspect(self.partial_aggregate, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate)
        self.inspector.inspect(self.update, pop_first=True)

    def supports_fused_format(self):
        return self.inspector.implements('message_and_aggregate')

    def supports_sparse_format(self):
        return (self.inspector.implements('message')
                and (self.inspector.implements('aggregate') or self.aggr))

    def supports_partial_format(self):
        return (self.inspector.implements('partial_message') and
                (self.inspector.implements('partial_aggregate') or self.aggr))

    def format(self, adj_type: AdjType) -> Tuple[str, str]:
        # adj_formats = ['edge_index', 'sparse_adj', 'dense_adj']
        # mp_formats = ['fused', 'sparse', 'partial']

        adj_format = None
        if (torch.is_tensor(adj_type) and adj_type.size(0) == 2
                and adj_type.dtype == torch.long):
            adj_format = 'edge_index'
        elif isinstance(adj_type, SparseTensor):
            adj_format = 'sparse_adj'
        elif torch.is_tensor(adj_type):
            adj_format = 'dense_adj'

        if adj_format is None:
            return ValueError(
                ('Encountered an invalid object for `adj_type` in '
                 '`MessagePassing.propagate`. Supported types are (1) sparse '
                 'edge indices of type `torch.LongTensor` with shape '
                 '`[2, num_edges]`, (2) sparse adjacency matrices of type '
                 '`torch_sparse.SparseTensor`, or (3) dense adjacency '
                 'matrices of type `torch.Tensor`.'))

        mp_format = None

        # `edge_index` only support "tradional" message passing.
        if adj_format == 'edge_index':
            mp_format = 'sparse'

        # Set to user-desired format (in case it is given).
        elif self.format is not None:
            mp_format = self.format

        # Always choose `fused` if applicable.
        elif self.supports_fused_format():
            mp_format = 'fused'

        # We prefer sparse format over the partial format for sparse adjacency
        # matrices since it is faster in general. We therefore only default to
        # `partial` mode if the user wants to implement some fancy customized
        # aggregations.
        elif adj_format == 'sparse_adj' and self.supports_sparse_format():
            mp_format = 'sparse'
        elif adj_format == 'sparse_adj' and self.supports_partial_format():
            mp_format = 'partial'

        # For dense message passing, we *require* partial aggregation.
        elif adj_format == 'dense_adj' and self.supports_partial_format():
            mp_format = 'partial'

        if mp_format is None:
            return ValueError(
                (f'Could not detect a valid message passing implementation '
                 f'for adjacency format "{adj_format}".'))

        return adj_format, mp_format

    def propagate(self, adj_type: AdjType, size: Optional[Tuple[int]] = None,
                  **kwargs) -> torch.Tensor:

        adj_format, mp_format = self.format(adj_type)

        if ((adj_format == 'sparse_adj' or adj_format == 'dense_adj')
                and self.flow == 'target_to_source'):
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'passing based on adjacency matrices. If you really want to '
                 'make use of reverse message passing flow, pass in the '
                 'transposed adjacency matrix to the message passing module, '
                 'e.g., `adj_t.t()`.'))

        if adj_format == 'edge_index':
            size = [None, None] if size is None else size
            size = size.tolist() if torch.is_tensor(size) else size
            size = list(size)
            assert len(size) == 2
        elif adj_format == 'sparse_adj':
            size = list(adj_type.sparse_sizes())[::-1]
        elif adj_format == 'dense_adj':
            size = list(adj_type.sizes())[:-2:-1]

        # We collect all arguments used for message passing in `data`.
        data = self.__collect__(adj_format, mp_format, adj_type, size, kwargs)

        pass

    def message(self) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    def partial_message(self) -> torch.Tensor:
        raise NotImplementedError

    def partial_aggregate(self, inputs) -> torch.Tensor:
        raise NotImplementedError

    def message_and_aggregate(self) -> torch.Tensor:
        raise NotImplementedError

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def check_consistency(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
