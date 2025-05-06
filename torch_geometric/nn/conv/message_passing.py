import os.path as osp
import warnings
from abc import abstractmethod
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from torch_geometric import EdgeIndex, is_compiling
from torch_geometric.index import ptr2index
from torch_geometric.inspector import Inspector, Signature
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.template import module_from_template
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}
HookDict = OrderedDict[int, Callable]


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers.

    Message passing layers follow the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
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

    # Supports `message_and_aggregate` via `EdgeIndex`.
    # TODO Remove once migration is finished.
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = False

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ) -> None:
        super().__init__()

        if flow not in ['source_to_target', 'target_to_source']:
            raise ValueError(f"Expected 'flow' to be either 'source_to_target'"
                             f" or 'target_to_source' (got '{flow}')")

        # Cast `aggr` into a string representation for backward compatibility:
        self.aggr: Optional[Union[str, List[str]]]
        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))
        self.flow = flow
        self.node_dim = node_dim

        # Collect attribute names requested in message passing hooks:
        self.inspector = Inspector(self.__class__)
        self.inspector.inspect_signature(self.message)
        self.inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        self.inspector.inspect_signature(self.message_and_aggregate, [0])
        self.inspector.inspect_signature(self.update, exclude=[0])
        self.inspector.inspect_signature(self.edge_update)

        self._user_args: List[str] = self.inspector.get_flat_param_names(
            ['message', 'aggregate', 'update'], exclude=self.special_args)
        self._fused_user_args: List[str] = self.inspector.get_flat_param_names(
            ['message_and_aggregate', 'update'], exclude=self.special_args)
        self._edge_user_args: List[str] = self.inspector.get_param_names(
            'edge_update', exclude=self.special_args)

        # Support for "fused" message passing:
        self.fuse = self.inspector.implements('message_and_aggregate')
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # Hooks:
        self._propagate_forward_pre_hooks: HookDict = OrderedDict()
        self._propagate_forward_hooks: HookDict = OrderedDict()
        self._message_forward_pre_hooks: HookDict = OrderedDict()
        self._message_forward_hooks: HookDict = OrderedDict()
        self._aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._aggregate_forward_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_hooks: HookDict = OrderedDict()
        self._edge_update_forward_pre_hooks: HookDict = OrderedDict()
        self._edge_update_forward_hooks: HookDict = OrderedDict()

        # Set jittable `propagate` and `edge_updater` function templates:
        self._set_jittable_templates()

        # Explainability:
        self._explain: Optional[bool] = None
        self._edge_mask: Optional[Tensor] = None
        self._loop_mask: Optional[Tensor] = None
        self._apply_sigmoid: bool = True

        # Inference Decomposition:
        self._decomposed_layers = 1
        self.decomposed_layers = decomposed_layers

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        if self.aggr_module is not None:
            self.aggr_module.reset_parameters()

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self.inspector = data['inspector']
        self.fuse = data['fuse']
        self._set_jittable_templates()
        super().__setstate__(data)

    def __repr__(self) -> str:
        channels_repr = ''
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            channels_repr = f'{self.in_channels}, {self.out_channels}'
        elif hasattr(self, 'channels'):
            channels_repr = f'{self.channels}'
        return f'{self.__class__.__name__}({channels_repr})'

    # Utilities ###############################################################

    def _check_input(
        self,
        edge_index: Union[Tensor, SparseTensor],
        size: Optional[Tuple[Optional[int], Optional[int]]],
    ) -> List[Optional[int]]:

        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            return [edge_index.num_rows, edge_index.num_cols]

        if is_sparse(edge_index):
            if self.flow == 'target_to_source':
                raise ValueError(
                    'Flow direction "target_to_source" is invalid for '
                    'message propagation via `torch_sparse.SparseTensor` '
                    'or `torch.sparse.Tensor`. If you really want to make '
                    'use of a reverse message passing flow, pass in the '
                    'transposed sparse tensor to the message passing module, '
                    'e.g., `adj_t.t()`.')

            if isinstance(edge_index, SparseTensor):
                return [edge_index.size(1), edge_index.size(0)]
            return [edge_index.size(1), edge_index.size(0)]

        elif isinstance(edge_index, Tensor):
            int_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32,
                          torch.int64)

            if edge_index.dtype not in int_dtypes:
                raise ValueError(f"Expected 'edge_index' to be of integer "
                                 f"type (got '{edge_index.dtype}')")
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.dim()} dimensions)")
            if not torch.jit.is_tracing() and edge_index.size(0) != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.size(0)}')")

            return list(size) if size is not None else [None, None]

        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, `torch_sparse.SparseTensor` or '
            '`torch.sparse.Tensor` for argument `edge_index`.')

    def _set_size(
        self,
        size: List[Optional[int]],
        dim: int,
        src: Tensor,
    ) -> None:
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                f'Encountered tensor with size {src.size(self.node_dim)} in '
                f'dimension {self.node_dim}, but expected size {the_size}.')

    def _index_select(self, src: Tensor, index) -> Tensor:
        if torch.jit.is_scripting() or is_compiling():
            return src.index_select(self.node_dim, index)
        else:
            return self._index_select_safe(src, index)

    def _index_select_safe(self, src: Tensor, index: Tensor) -> Tensor:
        try:
            return src.index_select(self.node_dim, index)
        except (IndexError, RuntimeError) as e:
            if index.numel() > 0 and index.min() < 0:
                raise IndexError(
                    f"Found negative indices in 'edge_index' (got "
                    f"{index.min().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.") from e

            if (index.numel() > 0 and index.max() >= src.size(self.node_dim)):
                raise IndexError(
                    f"Found indices in 'edge_index' that are larger "
                    f"than {src.size(self.node_dim) - 1} (got "
                    f"{index.max().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.") from e

            raise e

    def _lift(
        self,
        src: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: int,
    ) -> Tensor:
        if not torch.jit.is_scripting() and is_torch_sparse_tensor(edge_index):
            assert dim == 0 or dim == 1
            if edge_index.layout == torch.sparse_coo:
                index = edge_index._indices()[1 - dim]
            elif edge_index.layout == torch.sparse_csr:
                if dim == 0:
                    index = edge_index.col_indices()
                else:
                    index = ptr2index(edge_index.crow_indices())
            elif edge_index.layout == torch.sparse_csc:
                if dim == 0:
                    index = ptr2index(edge_index.ccol_indices())
                else:
                    index = edge_index.row_indices()
            else:
                raise ValueError(f"Unsupported sparse tensor layout "
                                 f"(got '{edge_index.layout}')")
            return src.index_select(self.node_dim, index)

        elif isinstance(edge_index, Tensor):
            if torch.jit.is_scripting():  # Try/catch blocks are not supported.
                index = edge_index[dim]
                return src.index_select(self.node_dim, index)
            return self._index_select(src, edge_index[dim])

        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            if dim == 0:
                return src.index_select(self.node_dim, col)
            elif dim == 1:
                return src.index_select(self.node_dim, row)

        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, `torch_sparse.SparseTensor` '
            'or `torch.sparse.Tensor` for argument `edge_index`.')

    def _collect(
        self,
        args: Set[str],
        edge_index: Union[Tensor, SparseTensor],
        size: List[Optional[int]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:

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
                        self._set_size(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self._set_size(size, dim, data)
                    data = self._lift(data, edge_index, dim)

                out[arg] = data

        if is_torch_sparse_tensor(edge_index):
            indices, values = to_edge_index(edge_index)
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = indices[0]
            out['edge_index_j'] = indices[1]
            out['ptr'] = None  # TODO Get `rowptr` from CSR representation.
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = values
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = None if values.dim() == 1 else values
            if out.get('edge_type', None) is None:
                out['edge_type'] = values

        elif isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]

            out['ptr'] = None
            if isinstance(edge_index, EdgeIndex):
                if i == 0 and edge_index.is_sorted_by_row:
                    (out['ptr'], _), _ = edge_index.get_csr()
                elif i == 1 and edge_index.is_sorted_by_col:
                    (out['ptr'], _), _ = edge_index.get_csc()

        elif isinstance(edge_index, SparseTensor):
            row, col, value = edge_index.coo()
            rowptr, _, _ = edge_index.csr()

            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = row
            out['edge_index_j'] = col
            out['ptr'] = rowptr
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = value
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = value
            if out.get('edge_type', None) is None:
                out['edge_type'] = value

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    # Message Passing #########################################################

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Runs the forward pass of the module."""

    def propagate(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        fuse = False
        if self.fuse and not self.explain:
            if is_sparse(edge_index):
                fuse = True
            elif (not torch.jit.is_scripting()
                  and isinstance(edge_index, EdgeIndex)):
                if (self.SUPPORTS_FUSED_EDGE_INDEX
                        and edge_index.is_sorted_by_col):
                    fuse = True

        if fuse:
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
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

            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
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

                coll_dict = self._collect(self._user_args, edge_index,
                                          mutable_size, kwargs)

                msg_kwargs = self.inspector.collect_param_data(
                    'message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data(
                    'aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.collect_param_data(
                    'update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, mutable_size, kwargs), out)
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

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)

    @abstractmethod
    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`
        or a :obj:`torch.sparse.Tensor`.
        """
        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

    # Edge-level Updates ######################################################

    def edge_updater(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""The initial call to compute or update features for each edge in the
        graph.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :obj:`torch.Tensor`, a
                :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying graph
                connectivity/message passing flow.
                See :meth:`propagate` for more information.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to compute or update
                features for each edge in the graph.
        """
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._edge_user_args, edge_index,
                                  mutable_size, kwargs)

        edge_kwargs = self.inspector.collect_param_data(
            'edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    @abstractmethod
    def edge_update(self) -> Tensor:
        r"""Computes or updates features for each edge in the graph.
        This function can take any argument as input which was initially passed
        to :meth:`edge_updater`.
        Furthermore, tensors passed to :meth:`edge_updater` can be mapped to
        the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        raise NotImplementedError

    # Inference Decomposition #################################################

    @property
    def decomposed_layers(self) -> int:
        return self._decomposed_layers

    @decomposed_layers.setter
    def decomposed_layers(self, decomposed_layers: int) -> None:
        if torch.jit.is_scripting():
            raise ValueError("Inference decomposition of message passing "
                             "modules is only supported on the Python module")

        if decomposed_layers == self._decomposed_layers:
            return  # Abort early if nothing to do.

        self._decomposed_layers = decomposed_layers

        if decomposed_layers != 1:
            if hasattr(self.__class__, '_orig_propagate'):
                self.propagate = self.__class__._orig_propagate.__get__(
                    self, MessagePassing)

        elif self.explain is None or self.explain is False:
            if hasattr(self.__class__, '_jinja_propagate'):
                self.propagate = self.__class__._jinja_propagate.__get__(
                    self, MessagePassing)

    # Explainability ##########################################################

    @property
    def explain(self) -> Optional[bool]:
        return self._explain

    @explain.setter
    def explain(self, explain: Optional[bool]) -> None:
        if torch.jit.is_scripting():
            raise ValueError("Explainability of message passing modules "
                             "is only supported on the Python module")

        if explain == self._explain:
            return  # Abort early if nothing to do.

        self._explain = explain

        if explain is True:
            assert self.decomposed_layers == 1
            self.inspector.remove_signature(self.explain_message)
            self.inspector.inspect_signature(self.explain_message, exclude=[0])
            self._user_args = self.inspector.get_flat_param_names(
                funcs=['message', 'explain_message', 'aggregate', 'update'],
                exclude=self.special_args,
            )
            if hasattr(self.__class__, '_orig_propagate'):
                self.propagate = self.__class__._orig_propagate.__get__(
                    self, MessagePassing)
        else:
            self._user_args = self.inspector.get_flat_param_names(
                funcs=['message', 'aggregate', 'update'],
                exclude=self.special_args,
            )
            if self.decomposed_layers == 1:
                if hasattr(self.__class__, '_jinja_propagate'):
                    self.propagate = self.__class__._jinja_propagate.__get__(
                        self, MessagePassing)

    def explain_message(
        self,
        inputs: Tensor,
        dim_size: Optional[int],
    ) -> Tensor:
        # NOTE Replace this method in custom explainers per message-passing
        # layer to customize how messages shall be explained, e.g., via:
        # conv.explain_message = explain_message.__get__(conv, MessagePassing)
        # see stackoverflow.com: 394770/override-a-method-at-instance-level
        edge_mask = self._edge_mask

        if edge_mask is None:
            raise ValueError("Could not find a pre-defined 'edge_mask' "
                             "to explain. Did you forget to initialize it?")

        if self._apply_sigmoid:
            edge_mask = edge_mask.sigmoid()

        # Some ops add self-loops to `edge_index`. We need to do the same for
        # `edge_mask` (but do not train these entries).
        if inputs.size(self.node_dim) != edge_mask.size(0):
            assert dim_size is not None
            edge_mask = edge_mask[self._loop_mask]
            loop = edge_mask.new_ones(dim_size)
            edge_mask = torch.cat([edge_mask, loop], dim=0)
        assert inputs.size(self.node_dim) == edge_mask.size(0)

        size = [1] * inputs.dim()
        size[self.node_dim] = -1
        return inputs * edge_mask.view(size)

    # Hooks ###################################################################

    def register_propagate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
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

    def register_propagate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
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

    def register_message_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
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

    def register_aggregate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`aggregate` is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_pre_hooks)
        self._aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`aggregate` has computed
        an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_hooks)
        self._aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`message_and_aggregate`
        is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_pre_hooks)
        self._message_and_aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`message_and_aggregate`
        has computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_hooks)
        self._message_and_aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`edge_update` is
        invoked. See :meth:`register_propagate_forward_pre_hook` for more
        information.
        """
        handle = RemovableHandle(self._edge_update_forward_pre_hooks)
        self._edge_update_forward_pre_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`edge_update` has
        computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._edge_update_forward_hooks)
        self._edge_update_forward_hooks[handle.id] = hook
        return handle

    # TorchScript Support #####################################################

    def _set_jittable_templates(self, raise_on_error: bool = False) -> None:
        root_dir = osp.dirname(osp.realpath(__file__))
        jinja_prefix = f'{self.__module__}_{self.__class__.__name__}'
        # Optimize `propagate()` via `*.jinja` templates:
        if not self.propagate.__module__.startswith(jinja_prefix):
            try:
                if ('propagate' in self.__class__.__dict__
                        and self.__class__.__dict__['propagate']
                        != MessagePassing.propagate):
                    raise ValueError("Cannot compile custom 'propagate' "
                                     "method")

                module = module_from_template(
                    module_name=f'{jinja_prefix}_propagate',
                    template_path=osp.join(root_dir, 'propagate.jinja'),
                    tmp_dirname='message_passing',
                    # Keyword arguments:
                    modules=self.inspector._modules,
                    collect_name='collect',
                    signature=self._get_propagate_signature(),
                    collect_param_dict=self.inspector.get_flat_param_dict(
                        ['message', 'aggregate', 'update']),
                    message_args=self.inspector.get_param_names('message'),
                    aggregate_args=self.inspector.get_param_names('aggregate'),
                    message_and_aggregate_args=self.inspector.get_param_names(
                        'message_and_aggregate'),
                    update_args=self.inspector.get_param_names('update'),
                    fuse=self.fuse,
                )

                self.__class__._orig_propagate = self.__class__.propagate
                self.__class__._jinja_propagate = module.propagate

                self.__class__.propagate = module.propagate
                self.__class__.collect = module.collect
            except Exception as e:  # pragma: no cover
                if raise_on_error:
                    raise e
                self.__class__._orig_propagate = self.__class__.propagate
                self.__class__._jinja_propagate = self.__class__.propagate

        # Optimize `edge_updater()` via `*.jinja` templates (if implemented):
        if (self.inspector.implements('edge_update')
                and not self.edge_updater.__module__.startswith(jinja_prefix)):
            try:
                if ('edge_updater' in self.__class__.__dict__
                        and self.__class__.__dict__['edge_updater']
                        != MessagePassing.edge_updater):
                    raise ValueError("Cannot compile custom 'edge_updater' "
                                     "method")

                module = module_from_template(
                    module_name=f'{jinja_prefix}_edge_updater',
                    template_path=osp.join(root_dir, 'edge_updater.jinja'),
                    tmp_dirname='message_passing',
                    # Keyword arguments:
                    modules=self.inspector._modules,
                    collect_name='edge_collect',
                    signature=self._get_edge_updater_signature(),
                    collect_param_dict=self.inspector.get_param_dict(
                        'edge_update'),
                )

                self.__class__._orig_edge_updater = self.__class__.edge_updater
                self.__class__._jinja_edge_updater = module.edge_updater

                self.__class__.edge_updater = module.edge_updater
                self.__class__.edge_collect = module.edge_collect
            except Exception as e:  # pragma: no cover
                if raise_on_error:
                    raise e
                self.__class__._orig_edge_updater = self.__class__.edge_updater
                self.__class__._jinja_edge_updater = (
                    self.__class__.edge_updater)

    def _get_propagate_signature(self) -> Signature:
        param_dict = self.inspector.get_params_from_method_call(
            'propagate', exclude=[0, 'edge_index', 'size'])
        update_signature = self.inspector.get_signature('update')

        return Signature(
            param_dict=param_dict,
            return_type=update_signature.return_type,
            return_type_repr=update_signature.return_type_repr,
        )

    def _get_edge_updater_signature(self) -> Signature:
        param_dict = self.inspector.get_params_from_method_call(
            'edge_updater', exclude=[0, 'edge_index', 'size'])
        edge_update_signature = self.inspector.get_signature('edge_update')

        return Signature(
            param_dict=param_dict,
            return_type=edge_update_signature.return_type,
            return_type_repr=edge_update_signature.return_type_repr,
        )

    def jittable(self, typing: Optional[str] = None) -> 'MessagePassing':
        r"""Analyzes the :class:`MessagePassing` instance and produces a new
        jittable module that can be used in combination with
        :meth:`torch.jit.script`.

        .. note::
            :meth:`jittable` is deprecated and a no-op from :pyg:`PyG` 2.5
            onwards.
        """
        warnings.warn(f"'{self.__class__.__name__}.jittable' is deprecated "
                      f"and a no-op. Please remove its usage.")
        return self
