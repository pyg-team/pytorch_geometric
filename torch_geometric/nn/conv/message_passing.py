import re
import inspect
from collections import OrderedDict
from uuid import uuid1
from tempfile import NamedTemporaryFile
from importlib.util import spec_from_file_location, module_from_spec
import sys
from typing import List, Tuple, Optional

import torch
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from .utils.helpers import unsqueeze
from .utils.inspector import Inspector, get_type

msg_aggr_special_args = set([
    'adj_t',
])

msg_special_args = set([
    'edge_index_i',
    'edge_index_j',
    'size_i',
    'size_j',
])

aggr_special_args = set([
    'ptr',
    'index',
    'dim_size',
])

update_special_args = set([])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    def __init__(self, aggr: str = "add", flow: str = "source_to_target",
                 node_dim: int = 0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        # self.inspector.inspect(self.message_and_aggregate)
        self.inspector.inspect(self.update, pop_first=True)

        self.__msg_aggr_params__ = inspect.signature(
            self.message_and_aggregate).parameters
        self.__msg_aggr_params__ = OrderedDict(self.__msg_aggr_params__)

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_aggr_args = (set(self.__msg_aggr_params__.keys()) -
                         msg_aggr_special_args)
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__user_args__ = set().union(msg_aggr_args, msg_args, aggr_args,
                                         update_args)

        # Support for "fused" message passing.
        self.__fuse__ = True

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

        # Support for TorchScript.
        self.__record_propagate__ = False
        self.__records__ = None

    def __set_size__(self, size: List[Optional[int]], idx: int,
                     tensor: Optional[torch.Tensor]):
        if not isinstance(tensor, torch.Tensor):
            pass
        elif size[idx] is None:
            assert tensor is not None
            size[idx] = tensor.size(self.node_dim)
        else:
            the_size = size[idx]
            assert the_size is not None
            assert tensor is not None
            if the_size != tensor.size(self.node_dim):
                raise ValueError((f'Encountered node tensor with size '
                                  f'{tensor.size(self.node_dim)} '
                                  f'in dimension {self.node_dim}, '
                                  f'but expected size {size[idx]}.'))

    def __collect__(self, edge_index, size, mp_type, kwargs, user_args):
        collect_trace = None
        if self.__record_propagate__:
            collect_trace = {}
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        for arg in user_args:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
                if self.__record_propagate__:
                    collect_trace[arg] = [
                        type(out[arg]), '{0}_out = kwargs.{0}'.format(arg)
                    ]
            else:
                sizestr = ''
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)
                is_tuple_or_list = (isinstance(data, tuple)
                                    or isinstance(data, list))

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    if self.__record_propagate__:
                        collect_trace[arg] = [
                            type(out[arg]), '{0}_out = None'.format(arg)
                        ]
                    continue

                if is_tuple_or_list:
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    sizestr = '        self.__set_size__(size, {0}, kwargs.{1}[{0}])'.format(
                        1 - idx, arg[:-2])  # noqa
                    data = data[idx]

                if not isinstance(data, torch.Tensor):
                    out[arg] = data
                    outtype = type(data) if data is not None \
                        else Optional[torch.Tensor]
                    if self.__record_propagate__:
                        if is_tuple_or_list and data is not None:
                            collect_trace[arg] = [
                                outtype, '{0}_out = kwargs.{1}[{2}]'.format(
                                    arg, arg[:-2], idx)
                            ]  # noqa
                        elif data is not None:
                            collect_trace[arg] = [
                                outtype,
                                '{0}_out = kwargs.{1}'.format(arg, arg[:-2])
                            ]  # noqa
                        elif data is None and not is_tuple_or_list:
                            # we have to assume it's a tensor
                            collect_trace[arg] = [
                                outtype,
                                '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1} is not None:\n            temp_{0} = kwargs.{1}\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'
                                .format(arg, arg[:-2], idx)
                            ]  # noqa
                        elif data is None:  # we have to assume it's a tensor
                            collect_trace[arg] = [
                                outtype,
                                '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1}[{2}] is not None:\n            temp_{0} = kwargs.{1}[{2}]\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'
                                .format(arg, arg[:-2], idx)
                            ]  # noqa
                        collect_trace[arg][1] += '\n' + sizestr
                    continue

                self.__set_size__(size, idx, data)
                if not is_tuple_or_list:
                    sizestr = '        self.__set_size__(size, {0}, kwargs.{1})'.format(
                        idx, arg[:-2])  # noqa

                if mp_type == 'edge_index':
                    out[arg] = data.index_select(self.node_dim,
                                                 edge_index[idx])
                    if self.__record_propagate__:
                        if is_tuple_or_list:
                            collect_trace[arg] = [
                                Optional[torch.Tensor],
                                '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1}[{2}] is not None:\n            temp_{0} = kwargs.{1}[{2}]\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'
                                .format(arg, arg[:-2], idx)
                            ]  # noqa
                        else:
                            typeout = type(out[arg])
                            if typeout == Optional[torch.Tensor]:
                                collect_trace[arg] = [
                                    typeout,
                                    '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1} is not None:\n            temp{1} = kwargs.{1}\n            assert temp{1} is not None\n            temp_{0} = temp{1}\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'
                                    .format(arg, arg[:-2], idx)
                                ]  # noqa
                            else:
                                collect_trace[arg] = [
                                    typeout,
                                    '{0}_out = kwargs.{1}.index_select(self.node_dim, edge_index[{2}])'
                                    .format(arg, arg[:-2], idx)
                                ]  # noqa
                        collect_trace[arg][1] += '\n' + sizestr
                elif mp_type == 'adj_t' and idx == 1:
                    rowptr = edge_index.storage.rowptr()
                    rowptr = unsqueeze(rowptr, dim=0, length=self.node_dim)
                    out[arg] = gather_csr(data, rowptr)
                    if self.__record_propagate__:
                        collect_trace[arg] = [
                            type(out[arg]),
                            '{0}_out = gather_csr(kwargs.{1}, edge_index.storage.rowptr())'
                            .format(arg, arg[:-2])
                        ]  # noqa
                elif mp_type == 'adj_t' and idx == 0:
                    col = edge_index.storage.col()
                    out[arg] = data.index_select(self.node_dim, col)
                    if self.__record_propagate__:
                        collect_trace[arg] = [
                            type(out[arg]),
                            '{0}_out = kwargs.{1}.index_select(self.node_dim, edge_index.storage.colptr())'
                            .format(arg, arg[:-2])
                        ]  # noqa

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
            out['ptr'] = None
            if self.__record_propagate__:
                collect_trace['edge_index_i'] = \
                    [type(out['edge_index_i']),
                     'edge_index_i_out = edge_index[{0}]'.format(i)]
                collect_trace['edge_index_j'] = \
                    [type(out['edge_index_j']),
                     'edge_index_j_out = edge_index[{0}]'.format(j)]
                collect_trace['index'] = [
                    type(out['index']), 'index_out = edge_index_i_out'
                ]
                collect_trace['ptr'] = [torch.Tensor, 'ptr_out = None']
        elif mp_type == 'adj_t':
            out['adj_t'] = edge_index
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['index'] = edge_index.storage.row()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_attr'] = edge_index.storage.value()
            if self.__record_propagate__:
                collect_trace['adj_t'] = [
                    type(out['adj_t']), 'adj_t_out = edge_index'
                ]
                collect_trace['edge_index_i'] = \
                    [type(out['edge_index_i']),
                     'edge_index_i_out = edge_index.storage.row()']
                collect_trace['edge_index_j'] = \
                    [type(out['edge_index_j']),
                     'edge_index_j_out = edge_index.storage.col()']
                collect_trace['index'] = \
                    [type(out['index']),
                     'index_out = edge_index.storage.row()']
                collect_trace['ptr'] = \
                    [type(out['ptr']),
                     'ptr_out = edge_index.storage.rowptr()']
                collect_trace['edge_attr'] = \
                    [type(out['edge_attr']),
                     'edge_attr_out = edge_index.storage.value()']

        out['size_j'] = size[j]
        out['size_i'] = size[i]
        out['dim_size'] = out['size_i']
        if self.__record_propagate__:
            collect_trace['size_i'] = [
                int,
                'size[0] = size[1] if size[0] is None else size[0]\n        size[1] = size[0] if size[1] is None else size[1]\n        size_i_out = size[{0}]'
                .format(i)
            ]  # noqa
            collect_trace['size_j'] = [int, 'size_j_out = size[{0}]'.format(j)]
            collect_trace['dim_size'] = [int, 'dim_size_out = size_i_out']

        return out, collect_trace

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    @torch.jit._overload_method  # noqa: F811
    def __determine_type_and_size__(
        self,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]],
    ) -> Tuple[str, List[Optional[int]]]:
        pass

    @torch.jit._overload_method  # noqa: F811
    def __determine_type_and_size__(
        self,
        edge_index: SparseTensor,
        size: Optional[Tuple[int, int]],
    ) -> Tuple[str, List[Optional[int]]]:
        pass

    def __determine_type_and_size__(  # noqa: F811
        self,
        edge_index,
        size: Optional[Tuple[int, int]],
    ) -> Tuple[str, List[Optional[int]]]:

        mp_type: str = ''

        if isinstance(edge_index, torch.Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            mp_type = 'edge_index'
        elif isinstance(edge_index, SparseTensor):
            mp_type = 'adj_t'
        else:
            raise ValueError(
                ('`MessagePassing.propagate` only supports `torch.LongTensor` '
                 'of shape `[2, num_messages]` or `torch_sparse.SparseTensor` '
                 'for argument :obj:`edge_index`.'))

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))

        size_out: List[Optional[int]] = [None, None]
        if mp_type == 'edge_index' and size is not None:
            size_out[0], size_out[1] = size[0], size[1]
        elif isinstance(edge_index, SparseTensor):
            size_out[0] = edge_index.sparse_size(1)
            size_out[1] = edge_index.sparse_size(0)

        return mp_type, size_out

    # TODO: Use overload.
    def propagate(self, edge_index, size: Optional[Tuple[int, int]] = None,
                  **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                message propagation.
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
        mp_type, size = self.__determine_type_and_size__(edge_index, size)

        # We collect all arguments used for message passing in `kwargs`.
        prop_kwargs = kwargs
        kwargs, collect_trace = self.__collect__(edge_index, size, mp_type,
                                                 kwargs, self.__user_args__)

        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ and not self.__explain__:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            msg_aggr_kwargs = {k: type(v) for k, v in msg_aggr_kwargs.items()}
            # Currently not supported in jitted-mode.
            if not self.__record_propagate__:
                out = self.message_and_aggregate(**msg_aggr_kwargs)
                if out == NotImplemented:
                    self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or not self.__fuse__ or self.__explain__:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require separate a separate message and
            # aggregate procedure since this allows us to easily inject an
            # `edge_mask` into the message passing computation.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(0) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * edge_mask.view(-1, 1)

            aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
            out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        if self.__record_propagate__:
            self.__records__ = {
                'prop_kwargs': prop_kwargs,
                'collect_trace': collect_trace
            }

        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
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

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            ptr = unsqueeze(ptr, dim=0, length=self.node_dim)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def message_and_aggregate(self, adj_t: SparseTensor) -> torch.Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        return NotImplemented

    def update(self, inputs: torch.Tensor):
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs

    @torch.jit.unused
    def __create_jittable_class__(self, prop_kwargs, collect_trace):
        # Define a new base class that replaces the `propagate` method.

        uid = uuid1().hex

        # `__init__` substitution.
        init_def = str(inspect.signature(self.__class__.__init__))
        init_args = str(inspect.signature(self.__init__))
        init_args = re.sub(r'=(.*?),', ',', init_args)
        init_args = re.sub(r'=(.*?)\)', ')', init_args)

        # `propagate` substitution.
        prop_init = ', '.join([f'{key}={key}' for key in prop_kwargs.keys()])
        prop_types = [f'{k}: {get_type(v)}' for k, v in prop_kwargs.items()]

        # `__collect__` substitution.
        collect_body = [v[1] for v in collect_trace.values()]
        args = ['{0}={0}_out'.format(key) for key in collect_trace]
        collect_body += ['return Collect_{}({})'.format(uid, ', '.join(args))]
        collect_body = '\n'.join([' ' * 8 + x for x in collect_body])
        collector_tuple_def = self.inspector.to_named_tuple(f'Collect_{uid}')

        # `message`, `aggregate` and `update` header substitutions.
        def render_args(args):
            return ', '.join([f'{arg}=kwargs.{arg}' for arg in args])

        msg_args = render_args(self.inspector.keys(['message']))
        aggr_args = render_args(self.inspector.keys(['aggregate']))
        update_args = render_args(self.inspector.keys(['update']))

        jit_module_repr = propagate_jittable_string.format(
            uid=uid,
            module=self.__class__.__module__,
            clsname=self.__class__.__name__,
            init_def=init_def,
            init_args=init_args,
            in_kwargs=', '.join(prop_types),
            in_kwargs_types='\n'.join([' ' * 4 + a for a in prop_types]),
            in_kwargs_init=prop_init,
            collector_tuple_def=collector_tuple_def,
            collect_body=collect_body,
            msg_args=msg_args,
            aggr_args=aggr_args,
            update_args=update_args,
        )
        print(jit_module_repr)

        ftemp = NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.py',
                                   delete=False)
        ftemp.write(jit_module_repr)
        ftemp.close()
        modname = f'{self.__class__.__name__}Jittable_{uid}'
        spec = spec_from_file_location(modname, ftemp.name)
        mod = module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, f'{self.__class__.__name__}Jittable_{uid}')

    @torch.jit.unused
    def jittable(self, **kwargs):
        r"""Analyzes the :obj:`MessagePassing` module and produces a
        jittable module."""

        # Run a partial trace of `forward` to gather type information.
        self.__record_propagate__ = True

        with torch.no_grad():
            self.forward(**kwargs)

        prop_kwargs = self.__records__['prop_kwargs']
        collect_trace = self.__records__['collect_trace']

        self.__record_propagate__ = False
        self.__records__ = None

        return self.__create_jittable_class__(prop_kwargs, collect_trace)


propagate_jittable_string = """
from typing import List, Tuple, Optional, NamedTuple

import torch
import {module}

class InKwargs_{uid}(NamedTuple):
{in_kwargs_types}

{collector_tuple_def}

class {clsname}Jittable_{uid}({module}.{clsname}):
    def __init__{init_def}:
        super({clsname}Jittable_{uid}, self).__init__{init_args}

    def __collect__(self, edge_index, size: List[Optional[int]], mp_type: str,
                    kwargs: InKwargs_{uid}):
{collect_body}

    def propagate(self, edge_index, {in_kwargs},
                  size: Optional[Tuple[int, int]] = None):

        in_kwargs = InKwargs_{uid}({in_kwargs_init})

        mp_type, the_size = self.__determine_type_and_size__(edge_index, size)
        kwargs = self.__collect__(edge_index, the_size, mp_type, in_kwargs)

        out = self.message({msg_args})
        out = self.aggregate(out, {aggr_args})
        out = self.update(out, {update_args})
        return out
"""
