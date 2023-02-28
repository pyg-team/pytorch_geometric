import inspect
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Module, ModuleDict, Parameter
from torch_scatter import scatter, scatter_add
from utils import IntegerQuantizer

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    softmax,
)

# REQUIRED_QUANTIZER_KEYS = ["aggregate", "message", "update_q"]

# msg_special_args = set(
#     [
#         "edge_index",
#         "edge_index_i",
#         "edge_index_j",
#         "size",
#         "size_i",
#         "size_j",
#     ]
# )

# aggr_special_args = set(
#     [
#         "index",
#         "dim_size",
#     ]
# )

# update_special_args = set([])


class MessagePassingQuant(Module):
    """Modified from the PyTorch Geometric message passing class"""

    # This would now be getting the following parameters for intializing the quantizers.
    #  qypte, ste, momentum, percentile, sign_input, sample_prop

    def __init__(self, qypte, ste, momentum, percentile, sign_input,
                 sample_prop, aggr="add", flow="source_to_target", node_dim=0,
                 mp_quantizers=None):
        super(MessagePassingQuant, self).__init__()

        # Supported aggregation methods. Add PyG compatible methods
        self.qypte = qypte
        self.ste = ste
        self.momentum = momentum
        self.percentile = percentile
        self.sign_input = sign_input
        self.sample_prop = sample_prop

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        # Changed this as an attribute to the class and then we can import the quantisers for each layers
        # self.keys = ["aggregate", "message", "update_q"]
        self.aggr_quantizer = IntegerQuantizer(qypte, ste, momentum,
                                               percentile, sign_input,
                                               sample_prop)
        self.message_quantizer = IntegerQuantizer(qypte, ste, momentum,
                                                  percentile, sign_input,
                                                  sample_prop)
        self.update_quantizer = IntegerQuantizer(qypte, ste, momentum,
                                                 percentile, sign_input,
                                                 sample_prop)

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
            "edge_index",
            "edge_index_i",
            "edge_index_j",
            "size",
            "size_i",
            "size_j",
        ])
        aggr_args = set(self.__aggr_params__.keys()) - set(
            ["index", "dim_size"])
        update_args = set(self.__update_params__.keys()) - set([])

        self.__args__ = set().union(msg_args, aggr_args, update_args)

        # assert mp_quantizers is not None
        # self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        # self.mp_quantizers = ModuleDict()
        # for key in self.keys:
        # self.mp_quantizers[key] = self.mp_quant_fns[key]()
        self.aggr_quantizer = IntegerQuantizer(self.qypte, self.ste,
                                               self.momentum, self.percentile,
                                               self.sign_input,
                                               self.sample_prop)
        self.message_quantizer = IntegerQuantizer(self.qypte, self.ste,
                                                  self.momentum,
                                                  self.percentile,
                                                  self.sign_input,
                                                  self.sample_prop)
        self.update_quantizer = IntegerQuantizer(self.qypte, self.ste,
                                                 self.momentum,
                                                 self.percentile,
                                                 self.sign_input,
                                                 self.sample_prop)

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
        # All the key word arguements listed here
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message_quantizer(self.message(**msg_kwargs))

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.aggr_quantizer(self.aggregate(out, **aggr_kwargs))

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update_quantizer(self.update(out, **update_kwargs))

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size,
                  limit=10000):  # pragma: no cover
        out = scatter(src=inputs, index=index, dim=self.node_dim,
                      dim_size=dim_size, reduce=self.aggr)

        if self.aggr == "max":
            out[out < -limit] = 0
        elif self.aggr == "min":
            out[out > limit] = 0
        return out

    def update(self, inputs):  # pragma: no cover
        return inputs
