import torch
from torch.nn import Linear
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.nn.conv import MessagePassing


class MeshConv(MessagePassing):
    r"""The Mesh Convolution operator from the `"MeshCNN: A Network with an
    Edge" <https://arxiv.org/abs/1809.05910>`_ paper
    This class implements the dot product between a kernel k and a neighborhood
    of an edge feature e.
    .. math::
        `\mathbf{MeshConv(e)} = \mathbf{e} \cdot \mathbf{k}_0 +
                                \sum_{k=1}^4 {e}_j \cdot {k}_j`
    where :math:`\mathbf{e}` is an edge feature, :math: `e_{j}` is the feature
    of the j-th convolutional neighbor of :math:`\mathbf{e}`.
    In this setting the receptive field for an edge e is given by:
    :math: `\left(e_1, e_2, e_3, e_4 \right) =
            \left(|a-c|,a+c,|b-d|,b+d \right)`
    where :math: `\mathbf(a),\mathbf(b),\mathbf(c),\mathbf(d)` are the actual
    edge features of :math:`\mathbf{e}`.
    Args:
        in_channels (int): Number of input channels of edge features.
        out_channels (int): Number of output channels of edge features.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True):
        super(MeshConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.neighborhood_size = 5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        for i in range(self.neighborhood_size):
            setattr(self, 'Linear{}'.format(i),
                    Linear(in_features=self.in_channels,
                           out_features=out_channels,
                           bias=self.bias))

    def forward(self, edge_features, edge_index):
        """
        Applies MessagePassing by calculating the features and applying
        linear layers on the neighbors and 'add' aggregation to create mesh
        convolution operation.
        Args:
            edge_features: edge features Tensor (Batch x Features x Edges)
            edge_index: edge index Tensor with structure (2 x Edge connections)
                        For example [[0, 0, 1, 1, 2, 2],[1, 2, 0, 2, 0, 1]] -
                        means all edges 0,1,2 connected together.
                        Note: in this implementation each edge must have 4
                        connections, making the neighbourhood size to
                        be 5 edges. If an edge does not have 4 neighbours the
                        input must have padding of "-1"s to create a valid
                        neighbourhood.
        """
        # prepare edge features to propagate - batch flatten
        edge_features = edge_features.squeeze(-1)
        batch_size, feature_dim, num_features = edge_features.shape

        edge_features = self.batch_flatten_features(edge_features)
        num_features_with_padding = edge_features.shape[0] // batch_size

        # prepare edge indexes to propagate - batch flatten
        flat_edge_index = \
            self.batch_flatten_edge_index(edge_index,
                                          num_features,
                                          num_features_with_padding,
                                          edge_features.device)

        # propagate message along graph
        out_edge_features = self.propagate(flat_edge_index, x=edge_features)

        # prepare output features data
        padded_dim = edge_features.shape[0]
        out_edge_features = out_edge_features.view(batch_size,
                                                   padded_dim // batch_size,
                                                   -1)
        out_edge_features = out_edge_features[:, 1:, :]  # remove dummy feature

        out_edge_features = out_edge_features.permute(0, 2, 1).contiguous()
        out_edge_features = out_edge_features.unsqueeze(dim=3)
        return out_edge_features

    @staticmethod
    def batch_flatten_features(edge_features):
        padding = torch.zeros(
            (edge_features.shape[0], edge_features.shape[1], 1),
            requires_grad=True,
            device=edge_features.device)
        edge_features = torch.cat((padding, edge_features), dim=2)
        edge_features = edge_features.permute(0, 2, 1).contiguous()
        edge_features = edge_features.view(
            edge_features.shape[0] * edge_features.shape[1],
            edge_features.shape[2])
        return edge_features

    def batch_flatten_edge_index(self, edge_index, num_features,
                                 num_features_padded, device):
        edge_index_dim = edge_index.shape
        padding = torch.zeros(2, num_features * self.neighborhood_size -
                              edge_index_dim[2], device=device).to(
            edge_index.device)

        edge_index = torch.cat(
            [torch.cat((e, padding), dim=1).unsqueeze(0) for e in edge_index])
        edge_index = edge_index + 1

        flat_edge_index = torch.zeros(2, edge_index_dim[
            0] * num_features * self.neighborhood_size, device=device).long()
        flat_edge_index[0] = torch.cat(
            [e[0] + i * num_features_padded for i, e in enumerate(edge_index)])
        flat_edge_index[1] = torch.cat(
            [e[1] + i * num_features_padded for i, e in enumerate(edge_index)])
        return flat_edge_index

    def message(self, edge_index, x, x_i, x_j):
        # prepare neighborhood features
        if self.neighborhood_size == 5:
            e0 = x_j[0::self.neighborhood_size]
            e1 = x_j[1::self.neighborhood_size] + x_j[3::self.neighborhood_size
                                                  ]
            e2 = x_j[2::self.neighborhood_size] + x_j[4::self.neighborhood_size
                                                  ]
            e3 = torch.abs(x_j[3::self.neighborhood_size] -
                           x_j[1::self.neighborhood_size])
            e4 = torch.abs(x_j[4::self.neighborhood_size] -
                           x_j[2::self.neighborhood_size])
            features_neighborhood = torch.stack([e0, e1, e2, e3, e4], dim=2)
        else:
            print("neighborhood size of {} is not supported yet.".format(
                self.neighborhood_size))
            exit()

        # apply linear layers on neighborhood size
        features_neighborhood = torch.stack([getattr(self,
                                                     'Linear{}'.format(i))(
            features_neighborhood[:, :, i])
            for i in
            range(self.neighborhood_size)],
            dim=2)

        # prepare data for aggregation ('add' aggregation)
        features_neighborhood = features_neighborhood.permute(0, 2,
                                                              1).contiguous()
        features_neighborhood = \
            features_neighborhood.view(1, features_neighborhood.shape[0] *
                                       features_neighborhood.shape[1],
                                       features_neighborhood.shape[
                                           2]).contiguous()
        return features_neighborhood

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, bias={}, hood=5, ' \
               'aggr=add)'.format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.bias)

    def reset_parameters(self):
        for i in range(self.neighborhood_size):
            xavier_normal_(getattr(self, 'Linear{}'.format(i)).weight.data)
            if self.bias:
                constant_(getattr(self, 'Linear{}'.format(i)).bias.data, 0)
