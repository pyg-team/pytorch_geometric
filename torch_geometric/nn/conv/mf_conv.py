import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear, ModuleDict


class MFConv(MessagePassing):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-
    learning-molecular-fingerprints.pdf>`_ paper

    This implementation includes a number of improvements from the
    `"Deep Learning for the Life Sciences" <https://www.amazon.com/
    Deep-Learning-Life-Sciences-Microscopy/dp/1492039837>`_ book.

    See also the `Deepchem implementation of this layer.
    <https://github.com/deepchem/deepchem/blob/2.3.0/
    deepchem/models/layers.py#L31>`_

    .. math::
        \mathbf{x}^{\prime}_i =
        \mathbf{H}^{deg(\mathbf{x}_i)}_{self} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)}
        \mathbf{H}^{deg(\mathbf{x}_i)}_{rel} \mathbf{x}_j

    where :math:`\mathbf{H}^{deg(v)}_{self}` and
    :math:`\mathbf{H}^{deg(v)}_{rel}` are the self and neighborhood weight
    tensors for a node of degree :math:`v`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        min_degree (int, optional): The minimum node degree to consider when
            updating weights (default: :obj:`0`)
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        self_weights(bool, optional): If set to :obj:`False`, don't train
            :math:`\mathbf{H}^{deg(v)}_{self}` weights. (default: :obj:`True`)
        self_loops: If set to :obj:`True`, include self loops in the graph.
            (default: :obj:`False)
    """
    def __init__(self,
                 in_channels, out_channels,
                 bias=True, min_degree=0, max_degree=10,
                 self_weights=True, self_loops=False):
        # define weights for each degree
        super(MFConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.self_weights = self_weights
        self.self_loops = self_loops
        self.H_rel = ModuleDict(
            {str(i): Linear(in_channels, out_channels, bias=bias)
             for i in range(min_degree, max_degree)})
        if self_weights:
            self.H_self = ModuleDict(
                {str(i): Linear(in_channels, out_channels, bias=bias)
                 for i in range(min_degree, max_degree)})

    def reset_parameters(self):
        for H in self.H_rel.values():
            H.reset_parameters()
        if self.self_weights:
            for H in self.H_self.values():
                H.reset_parameters()

    def forward(self, x, edge_index):
        if self.self_loops:
            # self loops seems to be implied:
            # https://github.com/deepchem/deepchem/blob/2.3.0/deepchem/models/layers.py#L126 # noqa
            # However this will increase the degree of every node by one.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], x.size(0)).type_as(x)
        return self.propagate(
            edge_index=edge_index,
            size=(x.size(0), x.size(0)),
            x=x,
            deg=deg,
        )

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, deg, x):
        r = torch.zeros(
            size=(aggr_out.shape[0], self.out_channels)
        ).type_as(aggr_out)

        for i in deg.clamp(self.min_degree, self.max_degree - 1).unique():
            # loop over every degree found in the graph
            i = int(i.item())
            H_rel = self.H_rel[str(i)]
            # build a selector matrix to pick out node features
            # belonging to nodes with degree i
            selector = torch.diag(torch.as_tensor(deg == i).type_as(aggr_out))
            r += H_rel(selector @ aggr_out)
            if self.self_weights:
                H_self = self.H_self[str(i)]
                r += H_self(selector @ x)
        return torch.sigmoid(r)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
