
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, JumpingKnowledge
from basic_gnn_model import Basic_GNN_model, Basic_GNN_model_WithJK


class GIN(torch.nn.Module):
    r"""The graph model from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper using the
    :class:`torch_geometric.nn.GINConv` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GINConv layers.
        train_eps (bool): Whether the central node re-weighting coefficient is trainable (default: :obj:`true`)
    """
    def __init__(self, in_channels, hidden_channels, num_layers, train_eps: bool = True):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ), train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=train_eps))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GINWithJK(torch.nn.Module):
    r"""The graph model from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper using the
    :class:`torch_geometric.nn.GINConv` layer 
    and the using the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
        train_eps (bool): Whether the central node re-weighting coefficient is trainable (default: :obj:`true`)
    """
    def __init__(self, in_channels, hidden_channels, num_layers, mode: str ='cat', train_eps: bool = True):
        super(GINWithJK, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ), train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=train_eps))
        self.jump = JumpingKnowledge(mode)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GINE(GIN):
    r"""The graph model from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper using the :class:`torch_geometric.nn.GINEConv` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GINConv layers.
        train_eps (bool): Whether the central node re-weighting coefficient is trainable (default: :obj:`true`)
    """

    def __init__(self, in_channels, hidden_channels, num_layers, train_eps: bool = True):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ), train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=train_eps))

class GINEWithJK(GINWithJK):
    r"""The graph model from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper using the
    :class:`torch_geometric.nn.GINEConv` layer, 
    and the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
        train_eps (bool): Whether the central node re-weighting coefficient is trainable (default: :obj:`true`)
    """
    def __init__(self, in_channels, hidden_channels, num_layers, mode: str ='cat', train_eps: bool = True):
        super(GINEWithJK, self).__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ), train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=train_eps))
        self.jump = JumpingKnowledge(mode)
