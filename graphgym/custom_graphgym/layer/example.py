import torch
import torch.nn as nn
from torch.nn import Parameter

import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

####################



@register_layer('my_gcnconv')
class GCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer

    This one allows a custom edge_index to be passed in
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch, x, edge_index):
        batch.x = self.model(x, edge_index)
        return batch


####################

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output


# Example 1: Directly define a GraphGym format Conv
# take 'batch' as input and 'batch' as output
@register_layer('exampleconv1')
class ExampleConv1(MessagePassing):
    r"""Example GNN layer

    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, batch):
        """"""
        x, edge_index = batch.x, batch.edge_index
        x = torch.matmul(x, self.weight)

        batch.x = self.propagate(edge_index, x=x)

        return batch

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


# Example 2: First define a PyG format Conv layer
# Then wrap it to become GraphGym format
class ExampleConv2Layer(MessagePassing):
    r"""Example GNN layer

    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


@register_layer('exampleconv2')
class ExampleConv2(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super().__init__()
        self.model = ExampleConv2Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
