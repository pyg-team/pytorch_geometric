import inspect
from collections import OrderedDict
from uuid import uuid1
from tempfile import NamedTemporaryFile
from importlib.util import spec_from_file_location, module_from_spec
import sys
from typing import List, Optional, get_type_hints

import torch
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

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

msg_aggr_fused = """out = self.message_and_aggregate({msg_aggr_kwargs})
    if out == NotImplemented:
        self.__fuse__ = False"""

msg_aggr_sequential = """out = self.message({msg_kwargs})

        # turn off gnn explainer for now in jitted mode
        # if self.__gnn_explainer__:
        #     edge_mask = self.__edge_mask__.sigmoid()
        #     if out.size(0) != edge_mask.size(0):
        #         loop = edge_mask.new_ones(size[0])
        #         edge_mask = torch.cat([edge_mask, loop], dim=0)
        #     assert out.size(0) == edge_mask.size(0)
        #     out = out * edge_mask.view(-1, 1)

        out = self.aggregate(inputs=out, {aggr_kwargs})"""

propagate_jittable_string = """import torch
import {parent_module}
from typing import List, Tuple, Optional, NamedTuple

class InKwargs_{uid}(NamedTuple):
{in_kwargs_types}

class Kwargs_{uid}(NamedTuple):
{kwargs_types}

class MsgAggrKwargs_{uid}(NamedTuple):
{msg_aggr_kwargs_types}

class MsgKwargs_{uid}(NamedTuple):
{msg_kwargs_types}

class AggrKwargs_{uid}(NamedTuple):
{aggr_kwargs_types}

class {clsname}Jittable_{uid}({parent_module}.{clsname}):

    def __init__{unboundinitsig}:
        super({clsname}Jittable_{uid}, self).__init__{boundinitsig}

    def __collect__(self,
                    edge_index,
                    size: List[Optional[int]],
                    mp_type: str,
                    kwargs: InKwargs_{uid},
                    user_args: List[str]):
{rendered_collect}

    def propagate(self, edge_index,
                  {in_kwargs},
                  size: Optional[List[int]] = None):
        in_kwargs = InKwargs_{uid}({in_kwargs_init})
        user_args = {user_args_set}

        mp_type, thesize = \
self.__determine_type_and_size_jittable__(edge_index, size)
        kwargs = self.__collect__(edge_index,
                                  thesize,
                                  mp_type,
                                  in_kwargs,
                                  user_args)

        {msg_aggr_type}

        out = self.update(out, {update_args})

        return out

{clsname}Jittable_{uid}.propagate.__doc__ = \
{parent_module}.{clsname}.propagate.__doc__
"""


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

    def __init__(self, aggr="add", flow="source_to_target", node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

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

        self.__record_propagate = False
        self.__record_dict__ = None

        msg_aggr_args = (set(self.__msg_aggr_params__.keys())
                         - msg_aggr_special_args)
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__user_args__ = set().union(msg_aggr_args, msg_args, aggr_args,
                                         update_args)

        self.__fuse__ = True

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

    def __get_mp_type__(self, edge_index):
        if (isinstance(edge_index, torch.Tensor)
           and edge_index.dtype == torch.long
           and edge_index.dim() == 2
           and edge_index.size(0)):
            return 'edge_index'
        elif isinstance(edge_index, SparseTensor):
            return 'adj_t'
        else:
            raise ValueError(
                ('`MessagePassing.propagate` only supports `torch.LongTensor` '
                 'of shape `[2, num_messages]` or `torch_sparse.SparseTensor` '
                 'for argument :obj:`edge_index`.'))

    def __set_size__(self,
                     size: List[Optional[int]],
                     idx: int,
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
                raise ValueError(
                    (f'Encountered node tensor with size '
                     f'{tensor.size(self.node_dim)} '
                     f'in dimension {self.node_dim}, '
                     f'but expected size {size[idx]}.'))

    def __collect__(self, edge_index, size, mp_type, kwargs, user_args):
        collect_trace = None
        if self.__record_propagate:
            collect_trace = {}
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        for arg in user_args:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
                if self.__record_propagate:
                    collect_trace[arg] = [type(out[arg]),
                                          '{0}_out = kwargs.{0}'.format(arg)]
            else:
                sizestr = ''
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)
                is_tuple_or_list = (isinstance(data, tuple) or
                                    isinstance(data, list))

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    if self.__record_propagate:
                        collect_trace[arg] = [type(out[arg]),
                                              '{0}_out = None'.format(arg)]
                    continue

                if is_tuple_or_list:
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    sizestr = '        self.__set_size__(size, {0}, kwargs.{1}[{0}])'.format(1 - idx, arg[:-2]) # noqa
                    data = data[idx]

                if not isinstance(data, torch.Tensor):
                    out[arg] = data
                    outtype = type(data) if data is not None \
                        else Optional[torch.Tensor]
                    if self.__record_propagate:
                        if is_tuple_or_list and data is not None:
                            collect_trace[arg] = [outtype, '{0}_out = kwargs.{1}[{2}]'.format(arg, arg[:-2], idx)] # noqa
                        elif data is not None:
                            collect_trace[arg] = [outtype, '{0}_out = kwargs.{1}'.format(arg, arg[:-2])] # noqa
                        elif data is None and not is_tuple_or_list:
                            # we have to assume it's a tensor
                            collect_trace[arg] = [outtype, '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1} is not None:\n            temp_{0} = kwargs.{1}\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'.format(arg, arg[:-2], idx)] # noqa
                        elif data is None:  # we have to assume it's a tensor
                            collect_trace[arg] = [outtype, '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1}[{2}] is not None:\n            temp_{0} = kwargs.{1}[{2}]\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'.format(arg, arg[:-2], idx)] # noqa
                        collect_trace[arg][1] += '\n' + sizestr
                    continue

                self.__set_size__(size, idx, data)
                if not is_tuple_or_list:
                    sizestr = '        self.__set_size__(size, {0}, kwargs.{1})'.format(idx, arg[:-2]) # noqa

                if mp_type == 'edge_index':
                    out[arg] = data.index_select(self.node_dim,
                                                 edge_index[idx])
                    if self.__record_propagate:
                        if is_tuple_or_list:
                            collect_trace[arg] = [Optional[torch.Tensor], '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1}[{2}] is not None:\n            temp_{0} = kwargs.{1}[{2}]\n            assert temp_{0} is not None\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'.format(arg, arg[:-2], idx)] # noqa
                        else:
                            parent_arg = arg[:-2]
                            typeout = type(out[arg]) if parent_arg not in self.__fwd_hints__ else self.__fwd_hints__[parent_arg] # noqa
                            if typeout == Optional[torch.Tensor]:
                                collect_trace[arg] = [typeout, '{0}_out: Optional[torch.Tensor] = None\n        if kwargs.{1} is not None:\n            temp{1} = kwargs.{1}\n            assert temp{1} is not None\n            temp_{0} = temp{1}\n            {0}_out = temp_{0}.index_select(self.node_dim, edge_index[{2}])'.format(arg, arg[:-2], idx)] # noqa
                            else:
                                collect_trace[arg] = [typeout, '{0}_out = kwargs.{1}.index_select(self.node_dim, edge_index[{2}])'.format(arg, arg[:-2], idx)] # noqa
                        collect_trace[arg][1] += '\n' + sizestr
                elif mp_type == 'adj_t' and idx == 1:
                    rowptr = edge_index.storage.rowptr()
                    for _ in range(self.node_dim):
                        rowptr = rowptr.unsqueeze(0)
                    out[arg] = gather_csr(data, rowptr)
                    if self.__record_propagate:
                        collect_trace[arg] = [type(out[arg]), '{0}_out = gather_csr(kwargs.{1}, edge_index.storage.rowptr())'.format(arg, arg[:-2])] # noqa
                elif mp_type == 'adj_t' and idx == 0:
                    col = edge_index.storage.col()
                    out[arg] = data.index_select(self.node_dim, col)
                    if self.__record_propagate:
                        collect_trace[arg] = [type(out[arg]), '{0}_out = kwargs.{1}.index_select(self.node_dim, edge_index.storage.colptr())'.format(arg, arg[:-2])] # noqa

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
            out['ptr'] = None
            if self.__record_propagate:
                collect_trace['edge_index_i'] = \
                    [type(out['edge_index_i']),
                     'edge_index_i_out = edge_index[{0}]'.format(i)]
                collect_trace['edge_index_j'] = \
                    [type(out['edge_index_j']),
                     'edge_index_j_out = edge_index[{0}]'.format(j)]
                collect_trace['index'] = [type(out['index']),
                                          'index_out = edge_index_i_out']
                collect_trace['ptr'] = [torch.Tensor,
                                        'ptr_out = None']
        elif mp_type == 'adj_t':
            out['adj_t'] = edge_index
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['index'] = edge_index.storage.row()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_attr'] = edge_index.storage.value()
            if self.__record_propagate:
                collect_trace['adj_t'] = [type(out['adj_t']),
                                          'adj_t_out = edge_index']
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
        if self.__record_propagate:
            collect_trace['size_i'] = [int,
                                       'size[0] = size[1] if size[0] is None else size[0]\n        size[1] = size[0] if size[1] is None else size[1]\n        size_i_out = size[{0}]'.format(i)] # noqa
            collect_trace['size_j'] = [int,
                                       'size_j_out = size[{0}]'.format(j)]
            collect_trace['dim_size'] = [int,
                                         'dim_size_out = size_i_out']

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

    def __determine_type_and_size__(self, edge_index, size):
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))

        if mp_type == 'edge_index':
            if size is None:
                size = [None, None]
            elif isinstance(size, int):
                size = [size, size]
            elif torch.is_tensor(size):
                size = size.tolist()
            elif isinstance(size, tuple):
                size = list(size)
        elif mp_type == 'adj_t':
            size = list(edge_index.sparse_sizes())[::-1]

        assert isinstance(size, list)
        assert len(size) == 2

        return mp_type, size

    def __determine_type_and_size_jittable__(self,
                                             edge_index,
                                             size: Optional[List[int]]):
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))
        size_out: List[Optional[int]] = [None, None]
        if mp_type == 'edge_index':
            if size is None:
                size_out[0], size_out[1] = None, None
            elif isinstance(size, int):
                size_out[0], size_out[1] = size, size
            elif isinstance(size, torch.Tensor):
                assert len(size.size()) == 1
                assert len(size) == 2
                size_out[0], size_out[1] = size[0].item(), size[1].item()
            elif isinstance(size, tuple):
                assert len(size) == 2
                size_out[0], size_out[1] = size[0], size[1]
        elif mp_type == 'adj_t':
            raise ValueError('Cannot use \'adj_t\' from '
                             '\'torch_sparse\' in jitted mode!')

        assert isinstance(size_out, list)
        assert len(size_out) == 2

        return mp_type, size_out

    def propagate(self, edge_index, size=None, **kwargs):
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
                Hence, the only difference between those formats is that we
                need to input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix in case :obj:`edge_index` is a
                :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        mp_type, size = self.__determine_type_and_size__(edge_index, size)

        # We collect all arguments used for message passing in `kwargs`.
        kwargs_orig, kwargs_orig_types = None, None
        if self.__record_propagate:
            kwargs_orig = kwargs
            kwargs_orig_types = {k: type(v) for k, v in kwargs.items()}

        kwargs, collect_trace = self.__collect__(edge_index,
                                                 size,
                                                 mp_type,
                                                 kwargs,
                                                 self.__user_args__)

        out = None
        msg_aggr_kwargs, msg_kwargs, aggr_kwargs = None, None, None
        msg_aggr_kwargs_types, msg_kwargs_types, aggr_kwargs_types = \
            None, None, None
        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ and not self.__explain__:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            msg_aggr_kwargs = {k: type(v) for k, v in msg_aggr_kwargs.items()}
            if not self.__record_propagate:
                out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or not self.__fuse__ or self.__explain__:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            msg_kwargs_types = {k: type(v) for k, v in msg_kwargs.items()}
            if not self.__record_propagate or self.__full_eval__:
                out = self.message(**msg_kwargs)

            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(0) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * edge_mask.view(-1, 1)

            aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
            aggr_kwargs_types = {k: type(v) for k, v in aggr_kwargs.items()}
            if not self.__record_propagate or self.__full_eval__:
                out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        update_kwargs_types = {k: type(v) for k, v in update_kwargs.items()}
        if not self.__record_propagate or self.__full_eval__:
            out = self.update(out, **update_kwargs)

        if self.__record_propagate:
            self.__record_dict__ = \
                {'mp_type': mp_type,
                 'size': size,
                 'in_kwargs': kwargs_orig,
                 'in_kwargs_types': kwargs_orig_types,
                 'msg_aggr_kwargs': msg_aggr_kwargs,
                 'msg_aggr_kwargs_types': msg_aggr_kwargs_types,
                 'msg_kwargs': msg_kwargs,
                 'msg_kwargs_types': msg_kwargs_types,
                 'aggr_kwargs': aggr_kwargs,
                 'aggr_kwargs_types': aggr_kwargs_types,
                 'update_kwargs': update_kwargs,
                 'update_kwargs_types': update_kwargs_types,
                 'collect_trace': collect_trace}

        return out

    def message(self, x_j):
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

    def aggregate(self,
                  inputs,
                  index,
                  ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def message_and_aggregate(self, adj_t):
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """

        return NotImplemented

    def update(self, inputs):
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs

    @torch.jit.unused
    def __make_jittable_fcn__(self, fwd_hints, print_model):
        # define a new base class with the propagate method replaced
        uid = uuid1().hex
        clsname = self.__class__.__name__
        # write the new propagate function
        msg_aggr_type = None
        kwarg_templ = '{0}=kwargs.{0}'

        def render_kwargs(theargs):
            return [kwarg_templ.format(a) for a in theargs]

        if self.__record_dict__['msg_aggr_kwargs'] is not None:
            theargs = list(self.__record_dict__['msg_aggr_kwargs'].keys())
            theargs = render_kwargs(theargs)
            msg_aggr_type = msg_aggr_fused \
                .format(uid=uid, msg_aggr_kwargs=', '.join(theargs))
        elif (self.__record_dict__['msg_kwargs'] is not None and
              self.__record_dict__['aggr_kwargs'] is not None):
            msg_args = list(self.__record_dict__['msg_kwargs'].keys())
            msg_args = render_kwargs(msg_args)
            aggr_args = list(self.__record_dict__['aggr_kwargs'].keys())
            if self.__record_dict__['mp_type'] == 'edge_index':
                aggr_args.remove('ptr')
            aggr_args = render_kwargs(aggr_args)
            msg_aggr_type = msg_aggr_sequential \
                .format(uid=uid,
                        msg_kwargs=', '.join(msg_args),
                        aggr_kwargs=', '.join(aggr_args))
        else:
            raise Exception('No kwargs filled for model!')

        update_args = list(self.__record_dict__['update_kwargs'].keys())
        update_args = ['{0}=kwargs.{0}'.format(a) for a in update_args]

        in_kwargs = list(self.__record_dict__['in_kwargs'].keys())
        in_kwargs_init = \
            ', '.join(['{0}={0}'.format(arg) for arg in in_kwargs])
        in_kwargs_dict = ['\'{0}\': {0}'.format(a) for a in in_kwargs]
        in_kwargs = ['{0}'.format(a) for a in in_kwargs]
        in_kwargs_dict = '{' + ', '.join(in_kwargs_dict) + '}'

        user_args_set = ['\'{0}\''.format(uarg) for uarg in self.__user_args__]
        user_args_set = '[' + ', '.join(user_args_set) + ']'

        args_mapping = {'kwargs_types': {'edge_index_i': 'torch.Tensor',
                                         'edge_index_j': 'torch.Tensor',
                                         'index': 'torch.Tensor',
                                         'ptr': 'Optional[torch.Tensor]',
                                         'size_j': 'int',
                                         'size_i': 'int',
                                         'dim_size': 'int'}}

        collect_trace = self.__record_dict__['collect_trace']
        for k, v in self.__record_dict__.items():
            if 'types' in k:
                args_mapping[k] = {}
                if v is None:
                    continue
                for name, thetype in v.items():
                    qualtype = thetype.__name__
                    thetype = thetype
                    if name in fwd_hints:
                        thetype = fwd_hints[name]
                        if thetype == Optional[torch.Tensor]:
                            qualtype = 'Optional[torch.Tensor]'
                        else:
                            qualtype = thetype.__name__
                    elif name in collect_trace:
                        if collect_trace[name][0] == Optional[torch.Tensor]:
                            qualtype = 'Optional[torch.Tensor]'
                        else:
                            qualtype = collect_trace[name][0].__name__
                            thetype = collect_trace[name][0]
                            if thetype.__module__ != 'builtins':
                                qualtype = thetype.__module__ + '.' + qualtype
                    elif thetype is not Optional[torch.Tensor]:
                        if thetype.__module__ != 'builtins':
                            qualtype = thetype.__module__ + '.' + qualtype
                    if thetype == tuple:
                        kwname = k[:k.find('_type')]
                        thetuple = self.__record_dict__[kwname][name]
                        mytypes = []
                        for i in range(len(thetuple)):
                            if thetuple[i] is not None:
                                atype = type(thetuple[i])
                                atypestr = atype.__name__
                                if atype.__module__ != 'builtins':
                                    atypestr = \
                                        atype.__module__ + '.' + atypestr
                                mytypes.append('Optional[{0}]'
                                               .format(atypestr))
                            else:
                                mytypes.append('Optional[torch.Tensor]')
                        qualtype = 'Tuple[{0}]'.format(', '.join(mytypes))
                    if thetype == list:
                        pass  # fixme add in list stuff later
                    if name not in args_mapping[k].keys():
                        args_mapping[k][name] = '{0}'.format(qualtype)

        for k, v in self.__record_dict__['collect_trace'].items():
            qualtype = 'Optional[torch.Tensor]'
            if v[0] != Optional[torch.Tensor]:
                qualtype = v[0].__name__
                if v[0].__module__ != 'builtins':
                    qualtype = v[0].__module__ + '.' + qualtype
            if k in fwd_hints:
                qualtype = fwd_hints[k]
                if qualtype == Optional[torch.Tensor]:
                    qualtype = 'Optional[torch.Tensor]'
                else:
                    qualtype = qualtype.__name__
            if 'ptr' in k or 'size' in k:
                args_mapping['kwargs_types'][k] = \
                    'Optional[{0}]'.format(qualtype)
            else:
                args_mapping['kwargs_types'][k] = \
                    '{0}'.format(qualtype)

        rendered_collect = [v[1] for v in collect_trace.values()]
        collect_trace_outs = \
            ['{0}={0}_out'.format(key) for key in collect_trace]
        return_stmt = 'return Kwargs_{0}({1})' \
            .format(uid, ', '.join(collect_trace_outs))
        rendered_collect.append(return_stmt)
        rendered_collect = \
            '\n'.join(['        {0}'.format(ln) for ln in rendered_collect])

        in_kwargs_typed = ['{0}: {1}'.format(name, args_mapping['in_kwargs_types'][name]) for name in in_kwargs] # noqa

        init_def = str(inspect.signature(self.__class__.__init__))
        init_args = inspect.signature(self.__init__)
        init_args = '(' + ', '.join(list(init_args.parameters.keys())) \
            .replace(' kwargs', ' **kwargs') + ')'

        render_msg_aggr_kwargs_types = '\n'.join(['    %s: %s' % (k, v) for k, v in args_mapping['msg_aggr_kwargs_types'].items()]) # noqa
        render_msg_kwargs_types = '\n'.join(['    %s: %s' % (k, v) for k, v in args_mapping['msg_kwargs_types'].items()]) # noqa
        render_aggr_kwargs_types = '\n'.join(['    %s: %s' % (k, v) for k, v in args_mapping['aggr_kwargs_types'].items()]) # noqa

        propagate_string = propagate_jittable_string \
            .format(uid=uid,
                    clsname=clsname,
                    unboundinitsig=init_def,
                    boundinitsig=init_args,
                    parent_module=self.__class__.__module__,
                    in_kwargs=', '.join(in_kwargs_typed),
                    in_kwargs_dict=in_kwargs_dict,
                    in_kwargs_types='\n'.join(['    %s: %s' % (k, v) for k, v in args_mapping['in_kwargs_types'].items()]), # noqa
                    in_kwargs_init=in_kwargs_init,
                    kwargs_types='\n'.join(['    %s: %s' % (k, v) for k, v in args_mapping['kwargs_types'].items()]), # noqa
                    msg_aggr_kwargs_types=render_msg_aggr_kwargs_types if len(render_msg_aggr_kwargs_types) > 0 else '    pass', # noqa
                    msg_kwargs_types=render_msg_kwargs_types if len(render_msg_kwargs_types) > 0 else '    pass', # noqa
                    aggr_kwargs_types=render_aggr_kwargs_types if len(render_aggr_kwargs_types) > 0 else '    pass', # noqa
                    rendered_collect=rendered_collect,
                    user_args_set=user_args_set,
                    msg_aggr_type=msg_aggr_type,
                    update_args=', '.join(update_args))
        # create a temp file with our class in it
        ftemp = NamedTemporaryFile(mode='w+', encoding='utf-8',
                                   suffix='.py', delete=False)
        ftemp.write(propagate_string)
        ftemp.close()
        # create a python module out of it
        if print_model:
            print(propagate_string)
        modname = f'pyg_jit_{uid}'
        spec = spec_from_file_location(modname, ftemp.name)
        mod = module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        cls = getattr(mod, f'{clsname}Jittable_{uid}')

        return cls

    @torch.jit.unused
    def jittable(self,
                 print_model=False,
                 full_eval=False,
                 **kwargs):
        r"""Alters this this PyG module such that it is torch-jit-scriptable.
        The MessagePassing module is manifestly not jittable but can be
        analyzed to produce jittable modules. This function implements that
        analysis and synthesis.

        Args:
            print_model (bool, Optional): Print out the jitted version of
                the model. Default is False.
            full_eval (bool, Optional): Perform a full evaluation of
                propagating when tracing the module. Turn this on when you
                need to assign from propagate within forward. Default is
                False.
        """
        # run a partial trace of the model's forward() to get types
        fwd_hints = get_type_hints(self.forward)
        self.__record_propagate = True
        with torch.no_grad():
            self.__fwd_hints__ = fwd_hints
            self.__full_eval__ = full_eval
            self.forward(**kwargs)
        self.__record_propagate = False

        out = self.__make_jittable_fcn__(fwd_hints, print_model)
        self.__record_dict__ = None
        self.__fwd_hints__ = None
        self.__full_eval__ = None

        return out
