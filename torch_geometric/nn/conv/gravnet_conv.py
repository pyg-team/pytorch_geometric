import torch
from torch.nn import Linear
from torch_scatter import scatter, segment_csr
from torch_geometric.nn.conv import MessagePassing

try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None


class GravNetConv(MessagePassing):
    r"""The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    The neighbors are constructed in a learnable low-dimensional projection of
    the feature space.
    A second projection of the input feature space is then propagated from the
    neighbors to each vertex using distance weights that are derived by
    applying a Gaussian function to the distances.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, space_dimensions,
                 propagate_dimensions, k, **kwargs):
        super(GravNetConv, self).__init__(**kwargs)

        if knn_graph is None:
            raise ImportError('`GravNetConv` requires `torch-cluster`.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.lin_s = Linear(in_channels, space_dimensions)
        self.lin_flr = Linear(in_channels, propagate_dimensions)
        self.lin_fout = Linear(in_channels + 2 * propagate_dimensions,
                               out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_flr.reset_parameters()
        self.lin_fout.reset_parameters()

    def forward(self, x, batch=None):
        """"""
        spatial = self.lin_s(x)
        to_propagate = self.lin_flr(x)

        edge_index = knn_graph(spatial, self.k, batch, loop=False,
                               flow=self.flow)

        reference = spatial.index_select(0, edge_index[1])
        neighbors = spatial.index_select(0, edge_index[0])

        distancessq = torch.sum((reference - neighbors)**2, dim=-1)
        # Factor 10 gives a better initial spread
        distance_weight = torch.exp(-10. * distancessq)

        prop_feat = self.propagate(edge_index, x=to_propagate,
                                   edge_weight=distance_weight)

        return self.lin_fout(torch.cat([prop_feat, x], dim=-1))

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            aggr_mean = segment_csr(inputs, ptr, reduce='mean')
            aggr_max = segment_csr(inputs, ptr, reduce='max')
        else:
            aggr_mean = scatter(inputs, index, dim=self.node_dim,
                                dim_size=dim_size, reduce='mean')
            aggr_max = scatter(inputs, index, dim=self.node_dim,
                               dim_size=dim_size, reduce='max')

        return torch.cat([aggr_mean, aggr_max], dim=-1)

    def __repr__(self):
        return '{}({}, {}, k={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.k)
