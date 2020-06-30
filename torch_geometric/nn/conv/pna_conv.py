import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn.modules import activation
from .utils.aggregators_and_scalers import AGGREGATORS, SCALERS, get_degree

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation=activation.ReLU(), dropout=0., b_norm=False, bias=True,
                 init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout, device=device)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = activation
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation=activation.ReLU(), last_activation=None,
                 dropout=0., mid_b_norm=False, last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregator from the `"Principal Neighbourhood Aggregation for Graph Nets"
        <https://arxiv.org/abs/2004.05718>`_ paper

        .. math::
            TODO

        where the general aggregation is performed

        .. math::
            TODO

        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggregators (str iterable): Set of aggregation function identifiers.
            scalers (str iterable): Set of scaling function identifiers.
            avg_d (float): The average in-degree of nodes in the training set.
            towers (int, optional): Number of towers to use.
                (default: :obj:`1`)
            pretrans_layers (int, optional): Number of layers in the transformation before the aggregation.
                (default: :obj:`1`)
            posttrans_layers (int, optional): Number of layers in the transformation after the aggregation.
                (default: :obj:`1`)
            divide_input (bool, optional): Whether the input features should be split between towers or not.
                (default: :obj:`False`)
            edge_features (bool, optional): Whether to consider the edge features. If set to :obj:`True`,
                a non-zero number of edge fetuares is expected.
                (default: :obj:`True`)

        """

    def __init__(self, in_channels, out_channels, aggregators, scalers, avg_d, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=False, edge_features=True, **kwargs):

        super(PNAConv, self).__init__(aggr=None, **kwargs)
        assert ((
                    not divide_input) or in_channels % towers == 0), "if divide_input is set the number of towers has to divide in_features"
        assert (out_channels % towers == 0), "the number of towers has to divide the out_features"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.towers = towers
        self.divide_input = divide_input
        self.edge_features = edge_features
        self.input_tower = self.in_channels // towers if divide_input else self.in_channels
        self.output_tower = self.out_channels // towers

        # retrieve the aggregators and scalers functions
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.avg_d = avg_d
        self.edge_encoder = FCLayer(in_size=in_channels, out_size=self.input_tower,
                                    activation=None) if edge_features else None

        # build pre-transformations and post-transformation MLP for each tower
        self.pretrans = nn.ModuleList()
        self.posttrans = nn.ModuleList()
        for _ in range(towers):
            self.pretrans.append(
                MLP(in_size=(3 if edge_features else 2) * self.input_tower, hidden_size=self.input_tower,
                    out_size=self.input_tower,
                    layers=pretrans_layers, mid_activation=activation.ReLU(), last_activation=None))
            self.posttrans.append(
                MLP(in_size=(len(self.aggregators) * len(self.scalers) + 1) * self.input_tower,
                    hidden_size=self.output_tower,
                    out_size=self.output_tower, layers=posttrans_layers, mid_activation=activation.ReLU(),
                    last_activation=None))

        self.mixing_network = FCLayer(self.out_channels, self.out_channels, activation=activation.LeakyReLU())

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=self.edge_encoder(edge_attr) if self.edge_features else None)

    def message(self, x_i, x_j, edge_attr):
        if self.divide_input:
            # divide the features among the towers
            x_i = x_i.view(-1, self.towers, self.input_tower)
            x_j = x_j.view(-1, self.towers, self.input_tower)
        else:
            # repeat the features over the towers
            x_i = x_i.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)
            x_j = x_j.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)

        if self.edge_features:
            edge_attr = edge_attr.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)

        # pre-transformation
        h_cat = torch.cat([x_i, x_j, edge_attr] if edge_attr else [x_i, x_j], dim=-1)
        y = torch.zeros((h_cat.shape[0], self.towers, self.input_tower), device=x_i.device)
        for tower, trans in enumerate(self.pretrans):
            y[:, tower, :] = trans(h_cat[:, tower, :])
        return y

    def aggregate(self, inputs, index, dim_size=None):
        D = get_degree(inputs, index, 0, dim_size)

        # aggregators
        inputs = torch.cat([aggregator(inputs, index, dim=0, dim_size=dim_size)
                            for aggregator in self.aggregators], dim=-1)
        # scalers
        return torch.cat([scaler(inputs, D, self.avg_d) for scaler in self.scalers], dim=-1)

    def update(self, aggr_out, x):
        # post-transformation
        if self.divide_input:
            x = x.view(-1, self.towers, self.input_tower)
        else:
            x = x.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)
        aggr_cat = torch.cat([x, aggr_out], dim=-1)
        y = torch.zeros((aggr_cat.shape[0], self.towers, self.output_tower), device=x.device)
        for tower, trans in enumerate(self.posttrans):
            y[:, tower, :] = trans(aggr_cat[:, tower, :])

        # concatenate and mix all the towers
        y = y.view(-1, self.towers * self.output_tower)
        y = self.mixing_network(y)

        return y
