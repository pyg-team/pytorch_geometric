import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing

from models.pytorch_geometric.aggregators import AGGREGATORS
from models.pytorch_geometric.scalers import get_degree, SCALERS
from models.layers import MLP, FCLayer

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class PNAConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggregators, scalers, avg_d, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=False, **kwargs):
        r"""
        :param in_channels:         size of the input per node
        :param in_channels:         size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param towers:              number of towers to use
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param divide_input:        whether the input features should be split between towers or not
        """
        super(PNAConv, self).__init__(aggr=None, **kwargs)
        assert ((not divide_input) or in_channels % towers == 0), "if divide_input is set the number of towers has to divide in_features"
        assert (out_channels % towers == 0), "the number of towers has to divide the out_features"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.towers = towers
        self.divide_input = divide_input
        self.input_tower = self.in_channels // towers if divide_input else self.in_channels
        self.output_tower = self.out_channels // towers

        # retrieve the aggregators and scalers functions
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.avg_d = avg_d

        self.edge_encoder = FCLayer(in_size=in_channels, out_size=self.input_tower, activation='none')

        # build pre-transformations and post-transformation MLP for each tower
        self.pretrans = nn.ModuleList()
        self.posttrans = nn.ModuleList()
        for _ in range(towers):
            self.pretrans.append(
                MLP(in_size=3 * self.input_tower, hidden_size=self.input_tower, out_size=self.input_tower,
                    layers=pretrans_layers, mid_activation='relu', last_activation='none'))
            self.posttrans.append(
                MLP(in_size=(len(self.aggregators) * len(self.scalers) + 1) * self.input_tower, hidden_size=self.output_tower,
                    out_size=self.output_tower, layers=posttrans_layers, mid_activation='relu', last_activation='none'))

        self.mixing_network = FCLayer(self.out_channels, self.out_channels, activation='LeakyReLU')

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_i, x_j, edge_attr):
        if self.divide_input:
            # divide the features among the towers
            x_i = x_i.view(-1, self.towers, self.input_tower)
            x_j = x_j.view(-1, self.towers, self.input_tower)
        else:
            # repeat the features over the towers
            x_i = x_i.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)
            x_j = x_j.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)
        edge_attr = edge_attr.view(-1, 1, self.input_tower).repeat(1, self.towers, 1)

        # pre-transformation
        h_cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        y = torch.zeros((h_cat.shape[0], self.towers, self.input_tower), device=x_i.device)
        for tower, trans in enumerate(self.pretrans):
            y[:, tower, :] = trans(h_cat[:, tower, :])
        return y

    def aggregate(self, inputs, index, dim_size=None):
        D = get_degree(inputs, index, self.node_dim, dim_size)

        # aggregators
        inputs = torch.cat([aggregator(inputs, index, dim=self.node_dim, dim_size=dim_size)
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

