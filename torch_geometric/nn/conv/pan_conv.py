import sys
import inspect

import torch
torch.manual_seed(0)

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j', 'edge_weight'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


### build model

class PANConv(MessagePassing):
    def __init__(self, in_channels, out_channels, filter_size=3, filter_weight=None):
        super(PANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.filter_size = filter_size
        if filter_weight is None:
            self.filter_weight = torch.nn.Parameter(0.1 * torch.ones(filter_size), requires_grad=True)
            #self.filter_weight.to(self.lin.weight.get_device())

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Path integral
        num_nodes = x.shape[0]
        edge_index, edge_weight = self.panentropy(edge_index, num_nodes)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_index, size, edge_weight):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.mul(edge_weight).view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def panentropy(self, edge_index, num_nodes):

        # sparse to dense
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0, :], edge_index[1, :]] = 1

        # iteratively add weighted matrix power
        adjtmp = torch.eye(num_nodes, device=edge_index.device)
        pan_adj = self.filter_weight[0] * torch.eye(num_nodes, device=edge_index.device)

        for i in range(self.filter_size - 1):
            adjtmp = torch.mm(adjtmp, adj)
            pan_adj = pan_adj + self.filter_weight[i+1] * adjtmp

        # dense to sparse
        edge_index_new = torch.nonzero(pan_adj).t()
        edge_weight_new = pan_adj[edge_index_new[0], edge_index_new[1]]

        return edge_index_new, edge_weight_new
