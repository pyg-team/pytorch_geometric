
import torch 
from torch.nn import Parameter, ModuleDict
# from .messagepassing import MessagePassingQuant.
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from .messagepassing import *
class GCNConvQuant(MessagePassingQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        nn=None,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        mp_quantizers=None,
        layer_quantizers=None,
        **kwargs,
    ):
        super(GCNConvQuant, self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)
        self.nn  =nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in ["weights","inputs","features", "norm"]:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):

        # quantizing input
        x_q = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights"](self.weight)
        x = torch.matmul(x_q, w_q)

        x = self.layer_quantizers["features"](x)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight

            norm = self.layer_quantizers["norm"](norm)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
    
class GCNConvMultiQuant(MessagePassingMultiQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        nn=None,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        layer_quantizers=None,
        mp_quantizers=None,
        **kwargs,
    ):
        super(GCNConvMultiQuant, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn  = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
        self.nn.reset_parameters()
        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in ["weights_low","inputs_low","inputs_high","features_low","features_high","norm_low"]:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, mask, edge_weight=None):
        # quantizing input
        if self.training:
            x_q = torch.empty_like(x)
            x_q[mask] = self.layer_quantizers["inputs_high"](x[mask])
            x_q[~mask] = self.layer_quantizers["inputs_low"](x[~mask])
        else:
            x_q = self.layer_quantizers["inputs_low"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights_low"](self.weight)
        if self.training:
            x = torch.empty((x_q.shape[0], w_q.shape[1])).to(x_q.device)
            x_tmp = torch.matmul(x_q, w_q)
            x[mask] = self.layer_quantizers["features_high"](x_tmp[mask])
            x[~mask] = self.layer_quantizers["features_low"](x_tmp[~mask])
        else:
            x = self.layer_quantizers["features_low"](torch.matmul(x_q, w_q))

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight
            norm = self.layer_quantizers["norm_low"](norm)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm, mask=mask)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        return self.nn(aggr_out)
        
#         return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )