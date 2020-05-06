import inspect
from collections import OrderedDict
from uuid import uuid1
from tempfile import NamedTemporaryFile
from importlib.util import spec_from_file_location, module_from_spec
import sys
from typing import Dict, List

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

propagate_docstring = r"""The initial call to start propagating messages.

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

propagate_jittable_string = """
import {parent_module}

class {clsname}Jittable_{uid}({parent_module}.{clsname}):

    def propagate(self, edge_index, size=None, {in_kwargs}):
        in_kwargs = {in_kwargs_dict}
        user_args = {user_args_set}
        
        mp_type, size = self.__determine_type_and_size_jittable__(edge_index, size)
        kwargs = self.__collect_jittable__(edge_index, size, mp_type, in_kwargs, user_args)

        {msg_aggr_type}

        out = self.update(out, {update_args})

        return out
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

        msg_aggr_args = set(self.__msg_aggr_params__.keys()) - msg_aggr_special_args
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
        if (isinstance(edge_index, torch.Tensor) and edge_index.dtype == torch.long
                and edge_index.dim() == 2 and edge_index.size(0)):
            return 'edge_index'
        elif isinstance(edge_index, SparseTensor):
            return 'adj_t'
        else:
            raise ValueError(
                ('`MessagePassing.propagate` only supports `torch.LongTensor` '
                 'of shape `[2, num_messages]` or `torch_sparse.SparseTensor` '
                 'for argument :obj:`edge_index`.'))

    def __set_size__(self, size: List[int], idx: int, tensor):
        if not isinstance(tensor, torch.Tensor):
            pass
        elif size[idx] is None:
            size[idx] = tensor.size(self.node_dim)
        elif size[idx] != tensor.size(self.node_dim):
            raise ValueError(
                (f'Encountered node tensor with size '
                 f'{tensor.size(self.node_dim)} in dimension {self.node_dim}, '
                 f'but expected size {size[idx]}.'))

    def __collect__(self, edge_index, size, mp_type, kwargs, user_args):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        for arg in user_args:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not isinstance(data, torch.Tensor):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)

                if mp_type == 'edge_index':
                    out[arg] = data.index_select(self.node_dim,
                                                 edge_index[idx])
                elif mp_type == 'adj_t' and idx == 1:
                    rowptr = edge_index.storage.rowptr()
                    for _ in range(self.node_dim):
                        rowptr = rowptr.unsqueeze(0)
                    out[arg] = gather_csr(data, rowptr)
                elif mp_type == 'adj_t' and idx == 0:
                    col = edge_index.storage.col()
                    out[arg] = data.index_select(self.node_dim, col)

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
            out['ptr'] = None
        elif mp_type == 'adj_t':
            out['adj_t'] = edge_index
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['index'] = edge_index.storage.row()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_attr'] = edge_index.storage.value()

        out['size_j'] = size[j]
        out['size_i'] = size[i]
        out['dim_size'] = out['size_i']

        return out
        
    def __collect_jittable__(self, edge_index, size: List[int], mp_type: str, kwargs: Dict[str, torch.Tensor], user_args: List[str]):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        parm_empty = torch.tensor([])
        for arg in user_args:
            arg = str(arg)
            trailer = str(arg[-2:])
            if trailer not in ij:
                out[arg] = kwargs.get(arg, parm_empty)
            else:
                idx = ij[trailer]
                data = kwargs.get(arg[:-2], parm_empty)

                if data is parm_empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not isinstance(data, torch.Tensor):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)

                if mp_type == 'edge_index':
                    out[arg] = data.index_select(self.node_dim,
                                                 edge_index[idx])
                elif mp_type == 'adj_t':
                    raise ValueError('Cannot use \'adj_t\' from \'torch_sparse\' in jitted mode!')

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
        elif mp_type == 'adj_t':
            raise ValueError('Cannot use \'adj_t\' from \'torch_sparse\' in jitted mode!')

        out['size_j'] = torch.tensor(size[j])
        out['size_i'] = torch.tensor(size[i])
        out['dim_size'] = out['size_i']

        return out
    
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
        
    def __determine_type_and_size_jittable__(self, edge_index, size):
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                'propagation based on `torch_sparse.SparseTensor`. If you '
                'really want to make use of a reverse message passing flow, '
                'pass in the transposed sparse tensor to the message passing '
                'module, e.g., `adj.t()`.'))
        size_out = [0, 0]
        if mp_type == 'edge_index':
            if size is None:
                size_out[0], size_out[1] = [None, None]
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
            raise ValueError('Cannot use \'adj_t\' from \'torch_sparse\' in jitted mode!')

        assert isinstance(size_out, list)
        assert len(size_out) == 2

        return mp_type, size_out

    def propagate(self, edge_index, size=None, **kwargs):
        mp_type, size = self.__determine_type_and_size__(edge_index, size)

        # We collect all arguments used for message passing in `kwargs`.
        kwargs_orig = kwargs
        kwargs = self.__collect__(edge_index, size, mp_type, kwargs, self.__user_args__)

        print('kwags processed ->', kwargs.keys())

        msg_aggr_kwargs, msg_kwargs, aggr_kwargs = None, None, None
        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ and not self.__explain__:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or not self.__fuse__ or self.__explain__:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            out = self.message(**msg_kwargs)

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

        if self.__record_propagate:
            self.__record_dict__ = {'mp_type': mp_type,
                                    'size': size,
                                    'kwargs': kwargs_orig,
                                    'msg_aggr_kwargs': msg_aggr_kwargs,
                                    'msg_kwargs': msg_kwargs,
                                    'aggr_kwargs': aggr_kwargs,
                                    'update_kwargs': update_kwargs}

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

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        print(self.__class__.__name__, ptr, dim_size)
        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            dim_size_int = dim_size.item() if isinstance(dim_size, torch.Tensor) else dim_size
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size_int,
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
    def __make_jittable_fcn__(self):
        # define a new base class with the propagate method replaced
        uid = uuid1().hex
        clsname = self.__class__.__name__
        # write the new propagate function
        msg_aggr_type = None
        if self.__record_dict__['msg_aggr_kwargs'] is not None:
            theargs = list(self.__record_dict__['msg_aggr_kwargs'].keys())
            theargs = ['{0}=kwargs[\'{0}\']'.format(a) for a in theargs]
            msg_aggr_type = msg_aggr_fused.format(msg_aggr_kwargs=', '.join(theargs))
        elif (self.__record_dict__['msg_kwargs'] is not None and
              self.__record_dict__['aggr_kwargs'] is not None):
            msg_args = list(self.__record_dict__['msg_kwargs'].keys())
            msg_args = ['{0}=kwargs[\'{0}\']'.format(a) for a in msg_args]
            aggr_args = list(self.__record_dict__['aggr_kwargs'].keys())
            if self.__record_dict__['mp_type'] == 'edge_index':
                aggr_args.remove('ptr')
            print('aggr_kwargs --->', aggr_args)
            aggr_args = ['{0}=kwargs[\'{0}\']'.format(a) for a in aggr_args]
            msg_aggr_type = msg_aggr_sequential.format(msg_kwargs=', '.join(msg_args),
                                                       aggr_kwargs=', '.join(aggr_args))
        else:
            raise Exception('No kwargs filled for model!')

        update_args = list(self.__record_dict__['update_kwargs'].keys())
        update_args = ['{0}=kwargs[\'{0}\']'.format(a) for a in update_args]

        in_kwargs = list(self.__record_dict__['kwargs'].keys())
        in_kwargs_dict = ['\'{0}\': {0}'.format(a) for a in in_kwargs]
        in_kwargs = ['{0}=None'.format(a) for a in in_kwargs]
        in_kwargs_dict = '{' + ', '.join(in_kwargs_dict)+ '}'
        
        user_args_set = ['\'{0}\''.format(uarg) for uarg in self.__user_args__]
        user_args_set = '[' + ', '.join(user_args_set) + ']'
        
        propagate_string = propagate_jittable_string.format(uid=uid,
                                                            clsname=clsname,
                                                            parent_module=self.__class__.__module__,
                                                            in_kwargs=', '.join(in_kwargs),
                                                            in_kwargs_dict=in_kwargs_dict,
                                                            user_args_set=user_args_set,
                                                            msg_aggr_type=msg_aggr_type,
                                                            update_args=', '.join(update_args))
        # create a temp file with our class in it
        ftemp = NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.py', delete=False)
        ftemp.write(propagate_string)
        ftemp.close()
        #create a python module out of it
        print(ftemp.name)
        print(propagate_string)
        modname = f'pyg_jit_{uid}'
        spec = spec_from_file_location(modname, ftemp.name)
        mod = module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        cls = getattr(mod, f'{clsname}Jittable_{uid}')
        
        out = cls.__new__(cls)

        return out
    
    @torch.jit.unused
    def jittable(self, **kwargs):
        r"""Alters this this PyG module such that it is torch-jit-scriptable.
        The MessagePassing module is manifestly not jittable but can be analyzed
        to produce jittable modules. This function implements that analysis and
        synthesis."""
        
        print(self.__class__.__name__)
        print(self.__class__.__module__)
        print(self.__class__.__qualname__)
        print(self.__class__.__bases__)
        
        fwd_sig = inspect.signature(self.forward)
        print('forward signature:')
        print(fwd_sig.parameters)
        
        prop_sig = inspect.signature(self.propagate)
        print('propagate signature:')
        print(prop_sig.parameters)
        
        #initialize the model
        print('old forward()')
        self.__record_propagate = True
        with torch.no_grad():
            print(self.forward(**kwargs))
        self.__record_propagate = False
        
        print(self.__class__)
        
        print('all kwargs :',self.__record_dict__['kwargs'].keys())
        if self.__record_dict__['msg_aggr_kwargs'] is not None:
            print('msg_aggr :', self.__record_dict__['msg_aggr_kwargs'].keys())
        if self.__record_dict__['msg_kwargs'] is not None:
            print('msg      :', self.__record_dict__['msg_kwargs'].keys())
        if self.__record_dict__['aggr_kwargs'] is not None:
            print('aggr     :', self.__record_dict__['aggr_kwargs'].keys())
        if self.__record_dict__['update_kwargs'] is not None:
            print('update   :', self.__record_dict__['update_kwargs'].keys())

        
        out = self.__make_jittable_fcn__()
        out.__dict__.update(self.__dict__)
        
        
        print(inspect.getfile(out.__class__))
        print(out.__class__.__name__, type(out.__class__))
        
        from torch.jit.frontend import get_source_lines_and_file
        sourcelines, file_lineno, filename = get_source_lines_and_file(out.propagate, torch._C.ErrorReport.call_stack())
        print(sourcelines, file_lineno, filename)
        
        print('USER ARGS --->', out.__user_args__)
        
        print('unjittable')
        print(self)
        print(self.__class__.__bases__)
        print(self.propagate)
                
        print('jittable')
        print(out)
        print(out.__class__.__bases__)
        print(out.propagate)
        
        print(type(out))
        print(out)
        print(out.aggregate)
        print(out.forward)
        out = torch.jit.script(out)
                
        print('new forward()')
        with torch.no_grad():
            print(out.forward(**kwargs))
            
        return out

MessagePassing.propagate.__doc__ = propagate_docstring
