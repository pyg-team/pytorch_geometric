import re
import inspect
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

special_args = set([
    'adj_t', 'edge_index_i', 'edge_index_j', 'size_i', 'size_j', 'ptr',
    'index', 'dim_size'
])


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
        self.inspector.inspect(self.message_and_aggregate)
        self.inspector.inspect(self.update, pop_first=True)

        self.__user_args__ = self.inspector.keys().difference(special_args)

        # Support for "fused" message passing.
        self.__fuse__ = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

        # Support for TorchScript.
        self.__record_propagate__ = False
        self.__records__ = None

    @torch.jit._overload_method  # noqa: F811
    def __determine_type_and_size__(  # noqa: F811
        self,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]],
    ) -> Tuple[str, List[Optional[int]]]:
        pass

    @torch.jit._overload_method  # noqa: F811
    def __determine_type_and_size__(  # noqa: F811
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

    def __collect__(self, edge_index, size, mp_type, kwargs):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        out = {}
        for arg in self.__user_args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, (tuple, list)):
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
                    rowptr = unsqueeze(rowptr, dim=0, length=self.node_dim)
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

    def __trace_collect__(self, edge_index, size, mp_type, kwargs):
        lines = []

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}

        lines += [('i, j = (0, 1) if self.flow == "target_to_source"'
                   ' else (1, 0)')]
        lines += ['ij = {"_i": i, "_j": j}']

        for arg in self.__user_args__:
            if arg[-2:] not in ij.keys():
                lines += [f'{arg}_out = kwargs.{arg}']
            else:
                lines += [f'idx = ij["{arg[-2:]}"]']
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    lines += [f'{arg}_out = None']
                    continue

                lines += [f'data = kwargs.{arg[:-2]}']

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    lines += [('self.__set_size__(size, 1 - idx, '
                               'data[1 - idx])')]
                    lines += ['data = data[idx]']
                    data = data[idx]

                if not isinstance(data, torch.Tensor):
                    lines += [f'{arg}_out = data']
                    continue

                lines += ['self.__set_size__(size, idx, data)']

                if mp_type == 'edge_index':
                    lines += [(f'{arg}_out = data.index_select('
                               f'self.node_dim, edge_index[idx])')]
                elif mp_type == 'adj_t' and idx == 1:
                    lines += ['rowptr = edge_index.storage.rowptr()']
                    lines += [('rowptr = unsqueeze(rowptr, dim=0, '
                               'length=self.node_dim)')]
                    lines += [f'{arg}_out = gather_csr(data, rowptr)']
                elif mp_type == 'adj_t' and idx == 0:
                    lines += ['col = edge_index.storage.col()']
                    lines += [(f'{arg}_out = data.index_select('
                               f'self.node_dim, col)')]

        lines += ['size[0] = size[1] if size[0] is None else size[0]']
        lines += ['size[1] = size[0] if size[1] is None else size[1]']

        if mp_type == 'edge_index':
            lines += ['edge_index_j_out = edge_index[j]']
            lines += ['edge_index_i_out = edge_index[i]']
            lines += ['ptr_out: Optional[torch.Tensor] = None']
            lines += ['adj_t_out: Optional[SparseTensor] = None']
        elif mp_type == 'adj_t':
            lines += ['adj_t_out = edge_index']
            lines += ['edge_index_i_out = edge_index.storage.row()']
            lines += ['edge_index_j_out = edge_index.storage.col()']
            lines += ['ptr_out = edge_index.storage.rowptr()']
            lines += ['edge_attr_out = edge_index.storage.value()']

        lines += ['index_out = edge_index_i_out']

        lines += ['size_j_out = size[j]']
        lines += ['size_i_out = size[i]']
        lines += ['dim_size_out = size_i_out']

        return lines

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
        coll_dict = self.__collect__(edge_index, size, mp_type, kwargs)

        if mp_type == 'adj_t' and self.__fuse__ and not self.__explain__:
            msg_aggr_kwargs = self.__distribute__(
                self.inspector.params['message_and_aggregate'], coll_dict)
            out = self.message_and_aggregate(**msg_aggr_kwargs)

        # Otherwise, run both functions in separation.
        elif mp_type == 'edge_index' or not self.__fuse__ or self.__explain__:
            msg_kwargs = self.__distribute__(self.inspector.params['message'],
                                             coll_dict)
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

            aggr_kwargs = self.__distribute__(
                self.inspector.params['aggregate'], coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.inspector.params['update'],
                                            coll_dict)
        out = self.update(out, **update_kwargs)

        if self.__record_propagate__:
            lines = self.__trace_collect__(edge_index, size, mp_type, kwargs)
            self.__records__ = {
                'mp_type': type(edge_index).__name__,
                'prop_kwargs': kwargs,
                'traced_collect': lines,
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

    def message_and_aggregate(self,
                              adj_t: Optional[SparseTensor]) -> torch.Tensor:
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
    def __create_jittable_class__(self, mp_type, prop_kwargs, traced_collect):
        # Define a new base class that replaces the `propagate` method.

        uid = uuid1().hex

        # `__init__` substitution.
        init_def = str(inspect.signature(self.__class__.__init__))
        init_args = str(inspect.signature(self.__init__))
        init_args = re.sub(r'=(.*?),', ',', init_args)
        init_args = re.sub(r'=(.*?)\)', ')', init_args)

        # `propagate` substitution.
        prop_types = [f'{k}: {get_type(v)}' for k, v in prop_kwargs.items()]
        prop_tuple_def = f'class Propagate_{uid}(NamedTuple):\n'
        prop_tuple_def += '\n'.join([' ' * 4 + x for x in prop_types])
        prop_types = ', '.join(prop_types)
        prop_args = ', '.join([f'{key}={key}' for key in prop_kwargs.keys()])

        # `__collect__` substitution.
        collect_body = traced_collect
        args = ['{0}={0}_out'.format(key) for key in self.inspector.keys()]
        collect_body += ['return Collect_{}({})'.format(uid, ', '.join(args))]
        collect_body = '\n'.join([' ' * 8 + x for x in collect_body])
        collector_tuple_def = self.inspector.to_named_tuple(f'Collect_{uid}')

        # `message`, `aggregate` and `update` header substitutions.
        def render_args(args):
            return ', '.join([f'{arg}=kwargs.{arg}' for arg in args])

        msg_args = render_args(self.inspector.keys(['message']))
        aggr_args = render_args(self.inspector.keys(['aggregate']))
        msg_aggr_args = render_args(
            self.inspector.keys(['message_and_aggregate']))
        update_args = render_args(self.inspector.keys(['update']))

        jit_module_repr = propagate_jittable_string.format(
            uid=uid,
            module=self.__class__.__module__,
            clsname=self.__class__.__name__,
            init_def=init_def,
            init_args=init_args,
            mp_type=mp_type,
            prop_tuple_def=prop_tuple_def,
            prop_types=prop_types,
            prop_args=prop_args,
            collector_tuple_def=collector_tuple_def,
            collect_body=collect_body,
            msg_args=msg_args,
            aggr_args=aggr_args,
            msg_aggr_args=msg_aggr_args,
            update_args=update_args,
        )

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

        if self.__records__ is None:
            return self.__class__

        mp_type = self.__records__['mp_type']
        prop_kwargs = self.__records__['prop_kwargs']
        traced_collect = self.__records__['traced_collect']

        self.__record_propagate__ = False
        self.__records__ = None

        return self.__create_jittable_class__(mp_type, prop_kwargs,
                                              traced_collect)


propagate_jittable_string = """
from typing import List, Tuple, Optional, NamedTuple

import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor
import {module}

{prop_tuple_def}

{collector_tuple_def}

class {clsname}Jittable_{uid}({module}.{clsname}):
    def __init__{init_def}:
        super({clsname}Jittable_{uid}, self).__init__{init_args}

    def __collect__(self, edge_index: {mp_type},
                    size: List[Optional[int]], mp_type: str,
                    kwargs: Propagate_{uid}):
{collect_body}

    def propagate(self, edge_index: {mp_type}, {prop_types},
                  size: Optional[Tuple[int, int]] = None):

        in_kwargs = Propagate_{uid}({prop_args})

        mp_type, the_size = self.__determine_type_and_size__(edge_index, size)
        kwargs = self.__collect__(edge_index, the_size, mp_type, in_kwargs)

        out = self.message({msg_args})
        out = self.aggregate(out, {aggr_args})
        return self.update(out, {update_args})
"""
