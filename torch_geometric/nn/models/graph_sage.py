import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge
from basic_gnn_model import Basic_GNN_model, Basic_GNN_model_WithJK

class GraphSAGE(Basic_GNN_model):
    r"""The GraphSAGE model from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper using the
    :class:`torch_geometric.nn.SAGEConv` as a convolutional operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GINConv layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GraphSAGE, self).__init__(conv_layer=SAGEConv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers)


class GraphSAGEWithJK(Basic_GNN_model_WithJK):
    r"""The GraphSAGE model from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper using the
    :class:`torch_geometric.nn.SAGEConv` as a convolutional operator
    using the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, mode='cat'):
        super(GraphSAGEWithJK, self).__init__(conv_layer=SAGEConv, in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers, mode=mode)