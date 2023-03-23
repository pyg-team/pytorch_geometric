import inspect
from collections import OrderedDict

import torch
from torch.nn import Module, ModuleDict
from torch_scatter import scatter
from .utils import scatter_


# This layer does not takethe input as any other layer. Derictly appliead
class MessagePassingQuant(Module):
    """
    Adapted from the original PyG message Passing class. Additional parameter
    to pass the quantization functions for each parameter

    Args:
    aggr (str or [str] or Aggregation, optional): The aggregation scheme
        to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
        In addition, can be any
        :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
        that automatically resolves to it).
        If given as a list, will make use of multiple aggregations in which
        different outputs will get concatenated in the last dimension.
        If set to :obj:`None`, the :class:`MessagePassing` instantiation is
        expected to implement its own aggregation logic via
        :meth:`aggregate`. (default: :obj:`"add"`)

    flow (str, optional): The flow direction of message passing
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)

    node_dim (int, optional): The axis along which to propagate.
    (default: :obj:`-2`)

    mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for
    each Message Passing Layer parameter

    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

    """
    def __init__(self, aggr="add", flow="source_to_target", node_dim=0,
                 mp_quantizers=None):
        super(MessagePassingQuant, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - set([
            "edge_index", "edge_index_i", "edge_index_j", "size", "size_i",
            "size_j"
        ])
        aggr_args = set(self.__aggr_params__.keys()) - set([
            "index",
            "dim_size",
        ])
        update_args = set(self.__update_params__.keys()) - set([])
        self.__args__ = set().union(msg_args, aggr_args, update_args)

        assert mp_quantizers is not None
        self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        self.mp_quantizers = ModuleDict()
        for key in ["aggregate", "message", "update_q"]:
            self.mp_quantizers[key] = self.mp_quant_fns[key]()

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (f"Encountered node tensor with size "
                 f"{tensor.size(self.node_dim)} in dimension {self.node_dim}, "
                 f"but expected size {size[index]}."))

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
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

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.mp_quantizers["message"](self.message(**msg_kwargs))

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.mp_quantizers["aggregate"](self.aggregate(
            out, **aggr_kwargs))

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.mp_quantizers["update_q"](self.update(out, **update_kwargs))

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover

        return scatter_(inputs, index, self.node_dim, dim_size)

        out = scatter(src=inputs, index=index, dim=self.node_dim,
                      dim_size=dim_size, reduce=self.aggr)
        if self.aggr == "max":
            out[out < -10000] = 0
        elif self.aggr == "min":
            out[out > 10000] = 0
        return out

    def update(self, inputs):  # pragma: no cover
        return inputs


class MessagePassingMultiQuant(Module):
    """
    Adapted from the original PyG message Passing class. Updates the message
    propagation step using the quantized values This class uses the Degree
    Quant approach to quantize the respective inputs based on the mask.

    Args:
    aggr (str or [str] or Aggregation, optional): The aggregation scheme
        to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
        In addition, can be any
        :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
        that automatically resolves to it).
        If given as a list, will make use of multiple aggregations in which
        different outputs will get concatenated in the last dimension.
        If set to :obj:`None`, the :class:`MessagePassing` instantiation is
        expected to implement its own aggregation logic via
        :meth:`aggregate`. (default: :obj:`"add"`)

    flow (str, optional): The flow direction of message passing
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)

    node_dim (int, optional): The axis along which to propagate.
    (default: :obj:`-2`)

    mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for
    each Message Passing Layer parameter

    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

    """
    def __init__(
        self,
        aggr="add",
        flow="source_to_target",
        node_dim=0,
        mp_quantizers=None,
    ):

        super(MessagePassingMultiQuant, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - set([
            "edge_index", "edge_index_i", "edge_index_j", "size", "size_i",
            "size_j"
        ])
        aggr_args = set(self.__aggr_params__.keys()) - set([
            "index",
            "dim_size",
        ])
        update_args = set(self.__update_params__.keys()) - set([])

        self.__args__ = set().union(msg_args, aggr_args, update_args)

        assert mp_quantizers is not None
        self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        self.mp_quantizers = ModuleDict()
        for key in [
                "message_low", "message_high", "update_low", "update_high",
                "aggregate_low", "aggregate_high"
        ]:
            self.mp_quantizers[key] = self.mp_quant_fns[key]()

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (f"Encountered node tensor with size "
                 f"{tensor.size(self.node_dim)} in dimension {self.node_dim}, "
                 f"but expected size {size[index]}."))

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
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

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, mask, size=None, **kwargs):
        """
        Args:
            edge_index (torch.Tensor or SparseTensor): The tensor which is used
            to store the graph edges
            mask (torch.Tensor): The mask for the graph which is used to
            protect the nodes in the Degree Quant method

        """

        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        msg = self.message(**msg_kwargs)
        if self.training:
            edge_mask = torch.index_select(mask, 0, edge_index[0])
            out = torch.empty_like(msg)
            out[edge_mask] = self.mp_quantizers["message_high"](msg[edge_mask])
            out[~edge_mask] = self.mp_quantizers["message_low"](
                msg[~edge_mask])
        else:
            out = self.mp_quantizers["message_low"](msg)

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        aggrs = self.aggregate(out, **aggr_kwargs)
        if self.training:
            out = torch.empty_like(aggrs)
            out[mask] = self.mp_quantizers["aggregate_high"](aggrs[mask])
            out[~mask] = self.mp_quantizers["aggregate_low"](aggrs[~mask])
        else:
            out = self.mp_quantizers["aggregate_low"](aggrs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        updates = self.update(out, **update_kwargs)
        if self.training:
            out = torch.empty_like(updates)
            out[mask] = self.mp_quantizers["update_high"](updates[mask])
            out[~mask] = self.mp_quantizers["update_low"](updates[~mask])
        else:
            out = self.mp_quantizers["update_low"](updates)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover

        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

        out = scatter(src=inputs, index=index, dim=self.node_dim,
                      dim_size=dim_size, reduce=self.aggr)
        if self.aggr == "max":
            out[out < -10000] = 0
        elif self.aggr == "min":
            out[out > 10000] = 0
        return out

    def update(self, inputs):  # pragma: no cover
        return inputs
