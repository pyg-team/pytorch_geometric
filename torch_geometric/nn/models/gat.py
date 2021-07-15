import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, JumpingKnowledge
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, JumpingKnowledge
from basic_gnn_model import Basic_GNN_model, Basic_GNN_model_WithJK

class GAT(Basic_GNN_model):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper using the
    :class:`torch_geometric.nn.GATConv` as a convolutional operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GINConv layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, heads = 1):
        super(GAT, self).__init__(conv_layer=GATConv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers)
        self.conv1 = GATConv(in_channels = in_channels, out_channels = hidden_channels, heads = heads, concat=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(in_channels = hidden_channels, out_channels = hidden_channels, heads = heads, concat=False))


class GATWithJK(Basic_GNN_model_WithJK):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper using the
    :class:`torch_geometric.nn.GATConv` as a convolutional operator
    and using the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, mode='cat', heads = 1):
        super(GATWithJK, self).__init__(conv_layer=GATConv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers, mode=mode)
        self.conv1 = GATConv(in_channels = in_channels, out_channels = hidden_channels, heads = heads, concat=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(in_channels = hidden_channels, out_channels = hidden_channels, heads = heads, concat=False))
        self.jump = JumpingKnowledge(mode)

class GATv2(Basic_GNN_model):
    r"""The GAT model from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper using the
    :class:`torch_geometric.nn.GATv2Conv` as a convolutional operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GINConv layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, heads = 1):
        super(GAT, self).__init__(conv_layer=GATv2Conv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers)
        self.conv1 = GATv2Conv(in_channels = in_channels, out_channels = hidden_channels, heads = heads, concat=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = heads, concat=False))


class GATv2WithJK(Basic_GNN_model_WithJK):
    r"""The GATv2 model from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper using the
    :class:`torch_geometric.nn.GATv2Conv` as a convolutional operator
    and using the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, mode='cat', heads = 1):
        super(GATv2WithJK, self).__init__(conv_layer=GATv2Conv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers, mode=mode)
        self.conv1 = GATv2Conv(in_channels = in_channels, out_channels = hidden_channels, heads = heads, concat=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = heads, concat=False))
        self.jump = JumpingKnowledge(mode)

