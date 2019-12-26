import sys
import inspect

import torch
from torch_geometric.utils import scatter_

message_special_args = [
    "edge_index",
    "edge_index_i",
    "edge_index_j",
    "size",
    "size_i",
    "size_j",
]
aggregate_special_args = ["out", "index", "dim", "dim_size"]
update_special_args = ["aggr_out"]
__size_error_msg__ = (
    "All tensors which should get mapped to the same source "
    "or target nodes must be of same size in dimension 0."
)

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    __aggr__ = ("add", "mean", "max")

    def __init__(self, aggr="add", flow="source_to_target", node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in self.__aggr__

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__message_signature__ = inspect.signature(self.message)
        # skip self, out
        self.__update_signature__ = inspect.signature(self.update)
        if set(update_special_args) - set(self.__update_signature__.parameters):
            raise TypeError(
                "Incomplete signature of update: {} are missing required arguments".format(
                    set(update_special_args) - set(self.__update_signature__.parameters)
                )
            )
        self.__aggregate_signature__ = inspect.signature(self.aggregate)
        if set(aggregate_special_args) - set(self.__aggregate_signature__.parameters):
            raise TypeError(
                "Incomplete signature of aggregate: {} are missing required arguments".format(
                    set(aggregate_special_args)
                    - set(self.__aggregate_signature__.parameters)
                )
            )

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferred and assumed to be symmetric.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        dim = self.node_dim
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        message_parameters = dict()
        special_message_parameters = dict()
        for arg, param in self.__message_signature__.parameters.items():
            if arg in message_special_args:
                # require this loop to be finished
                special_message_parameters[arg] = param
            elif arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], param.default)
                if tmp is inspect._empty:
                    raise TypeError(
                        "Required parameter '{}' for message is empty".format(arg[:-2])
                    )
                elif tmp is None:  # pragma: no cover
                    message_parameters[arg] = tmp
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(dim)
                            if size[1 - idx] != tmp[1 - idx].size(dim):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if tmp is None:
                        message_parameters[arg] = tmp
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError(__size_error_msg__)

                        tmp = torch.index_select(tmp, dim, edge_index[idx])
                        message_parameters[arg] = tmp
            else:
                tmp = kwargs.get(arg, param.default)
                if tmp is inspect._empty:
                    raise TypeError(
                        "Required parameter '{}' for message is empty".format(arg)
                    )
                message_parameters[arg] = tmp

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs["edge_index"] = edge_index
        kwargs["size"] = size

        for (arg, param) in special_message_parameters.items():
            if arg[-2:] in ij.keys():
                message_parameters[arg] = kwargs[arg[:-2]][ij[arg[-2:]]]
            else:
                message_parameters[arg] = kwargs[arg]
        update_parameters = dict()
        for arg, param in self.__update_signature__.parameters.items():
            if arg in update_special_args:
                continue
            tmp = kwargs.get(arg, param.default)
            if tmp is inspect._empty:
                raise TypeError(
                    "Required parameter '{}' for update is empty".format(arg)
                )
            update_parameters[arg] = tmp
        aggregate_parameters = dict()
        for arg, param in self.__aggregate_signature__.parameters.items():
            tmp = kwargs.get(arg, param.default)
            if arg in aggregate_special_args:
                continue
            if tmp is inspect._empty:
                raise TypeError(
                    "Required parameter '{}' for update is empty".format(arg)
                )
            aggregate_parameters[arg] = tmp

        out = self.message(**message_parameters)
        out = self.aggregate(out, edge_index[i], dim, size[i], **aggregate_parameters)
        out = self.update(out, **update_parameters)

        return out

    def aggregate(self, out, index, dim, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbours as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter function that supports
        "mean", "sum", "max" operations. The choice of aggr is
        specified in :meth:`__init__` as :obj:`aggr` argument.

        Notes:
            :obj:`self`, :obj:`out`, :obj:`index`, :obj:`dim`,
            :obj:`dim_size` are required first args. However,
            you can request additional args to be passed.

        """
        out = scatter_(self.aggr, out, index, dim, dim_size)
        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out
