import inspect
import os
import os.path as osp
import re
from collections import OrderedDict
from inspect import Parameter
from itertools import chain
from typing import Callable, List, Optional, Set, Union, get_type_hints
from uuid import uuid1

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from torch_scatter import gather_csr, scatter, segment_csr
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, Size

from .utils.helpers import expand_left
from .utils.inspector import Inspector, func_body_repr, func_header_repr
from .utils.jit import class_from_module_repr
from .utils.typing import (
    parse_types,
    resolve_types,
    sanitize,
    split_types_repr,
)

AGGRS = {'add', 'sum', 'mean', 'min', 'max', 'mul'}


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string or list, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"` or :obj:`None`). If given as a list, will make use of
            multiple aggregations in which different outputs will get
            concatenated in the last dimension. (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(self, aggr: Optional[Union[str, List[str]]] = "add",
                 flow: str = "source_to_target", node_dim: int = -2,
                 decomposed_layers: int = 1):

        super().__init__()

        if aggr is None or isinstance(aggr, str):
            assert aggr is None or aggr in AGGRS
            self.aggr: Optional[str] = aggr
            self.aggrs: List[str] = []
        elif isinstance(aggr, (tuple, list)):
            assert len(set(aggr) | AGGRS) == len(AGGRS)
            self.aggr: Optional[str] = None
            self.aggrs: List[str] = aggr
        else:
            raise ValueError(f"Only strings, list and tuples are valid "
                             f"aggregation schemes (got '{type(aggr)}')")

        self.flow = flow
        assert flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        self.decomposed_layers = decomposed_layers

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.params['aggregate'].pop('aggr', None)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)
        self.inspector.inspect(self.edge_update)

        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate', 'update']).difference(self.special_args)
        self.__edge_user_args__ = self.inspector.keys(
            ['edge_update']).difference(self.special_args)

        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self._explain = False
        self._edge_mask = None
        self._loop_mask = None
        self._apply_sigmoid = True

        # Hooks:
        self._propagate_forward_pre_hooks = OrderedDict()
        self._propagate_forward_hooks = OrderedDict()
        self._message_forward_pre_hooks = OrderedDict()
        self._message_forward_hooks = OrderedDict()
        self._aggregate_forward_pre_hooks = OrderedDict()
        self._aggregate_forward_hooks = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks = OrderedDict()
        self._message_and_aggregate_forward_hooks = OrderedDict()
        self._edge_update_forward_pre_hooks = OrderedDict()
        self._edge_update_forward_hooks = OrderedDict()

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index, dim)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = edge_index.storage.value()
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = edge_index.storage.value()
            if out.get('edge_type', None) is None:
                out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self._explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain and len(self.aggrs) == 0):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            if decomposed_layers > 1:
                user_args = self.__user_args__
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self.__collect__(self.__user_args__, edge_index,
                                             size, kwargs)

                msg_kwargs = self.inspector.distribute('message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                # For `GNNExplainer`, we require a separate message and
                # aggregate procedure since this allows us to inject the
                # `edge_mask` into the message passing computation scheme.
                if self._explain:
                    edge_mask = self._edge_mask
                    if self._apply_sigmoid:
                        edge_mask = edge_mask.sigmoid()
                    # Some ops add self-loops to `edge_index`. We need to do
                    # the same for `edge_mask` (but do not train those).
                    if out.size(self.node_dim) != edge_mask.size(0):
                        edge_mask = edge_mask[self._loop_mask]
                        loop = edge_mask.new_ones(size[0])
                        edge_mask = torch.cat([edge_mask, loop], dim=0)
                    assert out.size(self.node_dim) == edge_mask.size(0)
                    out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

                aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                if len(self.aggrs) == 0:
                    out = self.aggregate(out, **aggr_kwargs)
                else:
                    outs = []
                    for aggr in self.aggrs:
                        tmp = self.aggregate(out, aggr=aggr, **aggr_kwargs)
                        outs.append(tmp)
                    out = self.combine(outs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.distribute('update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        r"""The initial call to compute or update features for each edge in the
        graph.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                See :meth:`propagate` for more information.
            **kwargs: Any additional data which is needed to compute or update
                features for each edge in the graph.
        """
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, kwargs))
            if res is not None:
                edge_index, kwargs = res

        size = self.__check_input__(edge_index, size=None)

        coll_dict = self.__collect__(self.__edge_user_args__, edge_index, size,
                                     kwargs)

        edge_kwargs = self.inspector.distribute('edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                  aggr: Optional[str] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean", "min", "max" and "mul" operations as
        specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        aggr = self.aggr if aggr is None else aggr
        assert aggr is not None
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=aggr)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def combine(self, inputs: List[Tensor]) -> Tensor:
        r"""Combines the outputs from multiple aggregations into a single
        representation. Will only get called in case :obj:`aggr` holds a list
        of aggregation schemes to use."""
        assert len(inputs) > 0
        return torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

    def edge_update(self) -> Tensor:
        r"""Computes or updates features for each edge in the graph.
        This function can take any argument as input which was initially passed
        to :meth:`edge_updater`.
        Furthermore, tensors passed to :meth:`edge_updater` can be mapped to
        the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        raise NotImplementedError

    def register_propagate_forward_pre_hook(self,
                                            hook: Callable) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`propagate` is invoked.
        It should have the following signature:

        .. code-block:: python

            hook(module, inputs) -> None or modified input

        The hook can modify the input.
        Input keyword arguments are passed to the hook as a dictionary in
        :obj:`inputs[-1]`.

        Returns a :class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the added hook by calling :obj:`handle.remove()`.
        """
        handle = RemovableHandle(self._propagate_forward_pre_hooks)
        self._propagate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_propagate_forward_hook(self,
                                        hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`propagate` has computed
        an output.
        It should have the following signature:

        .. code-block:: python

            hook(module, inputs, output) -> None or modified output

        The hook can modify the output.
        Input keyword arguments are passed to the hook as a dictionary in
        :obj:`inputs[-1]`.

        Returns a :class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the added hook by calling :obj:`handle.remove()`.
        """
        handle = RemovableHandle(self._propagate_forward_hooks)
        self._propagate_forward_hooks[handle.id] = hook
        return handle

    def register_message_forward_pre_hook(self,
                                          hook: Callable) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`message` is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._message_forward_pre_hooks)
        self._message_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_forward_hook(self, hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`message` has computed
        an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._message_forward_hooks)
        self._message_forward_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_pre_hook(self,
                                            hook: Callable) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`aggregate` is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_pre_hooks)
        self._aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_hook(self,
                                        hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`aggregate` has computed
        an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_hooks)
        self._aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_pre_hook(
            self, hook: Callable) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`message_and_aggregate`
        is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_pre_hooks)
        self._message_and_aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_hook(
            self, hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`message_and_aggregate`
        has computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_hooks)
        self._message_and_aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_pre_hook(
            self, hook: Callable) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`edge_update` is
        invoked. See :meth:`register_propagate_forward_pre_hook` for more
        information.
        """
        handle = RemovableHandle(self._edge_update_forward_pre_hooks)
        self._edge_update_forward_pre_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_hook(self,
                                          hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`edge_update` has
        computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._edge_update_forward_hooks)
        self._edge_update_forward_hooks[handle.id] = hook
        return handle

    @torch.jit.unused
    def jittable(self, typing: Optional[str] = None):
        r"""Analyzes the :class:`MessagePassing` instance and produces a new
        jittable module.

        Args:
            typing (string, optional): If given, will generate a concrete
                instance with :meth:`forward` types based on :obj:`typing`,
                *e.g.*: :obj:`"(Tensor, Optional[Tensor]) -> Tensor"`.
        """
        try:
            from jinja2 import Template
        except ImportError:
            raise ModuleNotFoundError(
                "No module named 'jinja2' found on this machine. "
                "Run 'pip install jinja2' to install the library.")

        source = inspect.getsource(self.__class__)

        # Find and parse `propagate()` types to format `{arg1: type1, ...}`.
        if hasattr(self, 'propagate_type'):
            prop_types = {
                k: sanitize(str(v))
                for k, v in self.propagate_type.items()
            }
        else:
            match = re.search(r'#\s*propagate_type:\s*\((.*)\)', source)
            if match is None:
                raise TypeError(
                    'TorchScript support requires the definition of the types '
                    'passed to `propagate()`. Please specify them via\n\n'
                    'propagate_type = {"arg1": type1, "arg2": type2, ... }\n\n'
                    'or via\n\n'
                    '# propagate_type: (arg1: type1, arg2: type2, ...)\n\n'
                    'inside the `MessagePassing` module.')
            prop_types = split_types_repr(match.group(1))
            prop_types = dict([re.split(r'\s*:\s*', t) for t in prop_types])

        # Find and parse `edge_updater` types to format `{arg1: type1, ...}`.
        if 'edge_update' in self.__class__.__dict__.keys():
            if hasattr(self, 'edge_updater_type'):
                edge_updater_types = {
                    k: sanitize(str(v))
                    for k, v in self.edge_updater.items()
                }
            else:
                match = re.search(r'#\s*edge_updater_type:\s*\((.*)\)', source)
                if match is None:
                    raise TypeError(
                        'TorchScript support requires the definition of the '
                        'types passed to `edge_updater()`. Please specify '
                        'them via\n\n edge_updater_type = {"arg1": type1, '
                        '"arg2": type2, ... }\n\n or via\n\n'
                        '# edge_updater_type: (arg1: type1, arg2: type2, ...)'
                        '\n\ninside the `MessagePassing` module.')
                edge_updater_types = split_types_repr(match.group(1))
                edge_updater_types = dict(
                    [re.split(r'\s*:\s*', t) for t in edge_updater_types])
        else:
            edge_updater_types = {}

        type_hints = get_type_hints(self.__class__.update)
        prop_return_type = type_hints.get('return', 'Tensor')
        if str(prop_return_type)[:6] == '<class':
            prop_return_type = prop_return_type.__name__

        type_hints = get_type_hints(self.__class__.edge_update)
        edge_updater_return_type = type_hints.get('return', 'Tensor')
        if str(edge_updater_return_type)[:6] == '<class':
            edge_updater_return_type = edge_updater_return_type.__name__

        # Parse `__collect__()` types to format `{arg:1, type1, ...}`.
        collect_types = self.inspector.types(
            ['message', 'aggregate', 'update'])

        # Parse `__collect__()` types to format `{arg:1, type1, ...}`,
        # specific to the argument used for edge updates.
        edge_collect_types = self.inspector.types(['edge_update'])

        # Collect `forward()` header, body and @overload types.
        forward_types = parse_types(self.forward)
        forward_types = [resolve_types(*types) for types in forward_types]
        forward_types = list(chain.from_iterable(forward_types))

        keep_annotation = len(forward_types) < 2
        forward_header = func_header_repr(self.forward, keep_annotation)
        forward_body = func_body_repr(self.forward, keep_annotation)

        if keep_annotation:
            forward_types = []
        elif typing is not None:
            forward_types = []
            forward_body = 8 * ' ' + f'# type: {typing}\n{forward_body}'

        root = os.path.dirname(osp.realpath(__file__))
        with open(osp.join(root, 'message_passing.jinja'), 'r') as f:
            template = Template(f.read())

        uid = uuid1().hex[:6]
        cls_name = f'{self.__class__.__name__}Jittable_{uid}'
        jit_module_repr = template.render(
            uid=uid,
            module=str(self.__class__.__module__),
            cls_name=cls_name,
            parent_cls_name=self.__class__.__name__,
            prop_types=prop_types,
            prop_return_type=prop_return_type,
            fuse=self.fuse,
            single_aggr=len(self.aggrs) == 0,
            collect_types=collect_types,
            user_args=self.__user_args__,
            edge_user_args=self.__edge_user_args__,
            forward_header=forward_header,
            forward_types=forward_types,
            forward_body=forward_body,
            msg_args=self.inspector.keys(['message']),
            aggr_args=self.inspector.keys(['aggregate']),
            msg_and_aggr_args=self.inspector.keys(['message_and_aggregate']),
            update_args=self.inspector.keys(['update']),
            edge_collect_types=edge_collect_types,
            edge_update_args=self.inspector.keys(['edge_update']),
            edge_updater_types=edge_updater_types,
            edge_updater_return_type=edge_updater_return_type,
            check_input=inspect.getsource(self.__check_input__)[:-1],
            lift=inspect.getsource(self.__lift__)[:-1],
        )
        # Instantiate a class from the rendered JIT module representation.
        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None
        return module

    def __repr__(self) -> str:
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels})')
        return f'{self.__class__.__name__}()'
