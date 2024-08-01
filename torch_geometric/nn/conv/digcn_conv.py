import scipy

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops
from typing import Optional
from torch_geometric.typing import Adj, OptTensor



def get_appr_directed_adj(alpha, edge_index : Adj, num_nodes : int, edge_weight=None):
    r"""
    Getting directed page rank based adjacency matrix.
    Digraph Inception Convolutional Networks
    `<https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper
    Args:
        alpha (float)-  Teleport probability in Page Rank.
        edge_index - Edge Indices in the given Graph.
        num_nodes (int) - Number of nodes in the given Graph.
        edge_weight (torch.Tensor) - attribute for the corresponding edge indices.  
    """
    if edge_weight ==None:
        edge_weight = torch.ones((edge_index.size(1), ))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_edge_weight =  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index,norm_edge_weight

def get_second_directed_adj(edge_index : Adj , num_nodes : int):
    r"""
    Getting additonal directed adjacency matrix,
    from Digraph Inception Convolutional Networks.
    `<https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper
    Args:
        edge_index - Edge Indices in the given Graph.
        num_nodes (int) - Number of nodes in the given Graph.
    """
    edge_weight = torch.ones((edge_index.size(1), ))
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0
    L = (L_in_hat + L_out_hat) / 2.0
    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]

    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 

    return edge_index,norm_edge_weight 

class DIGCNConv(MessagePassing):
    """
    Digraph Inception Convolutional Operator
    `<https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper
    uses :class:`torch_geometric.nn.conv.digcn_conv` operator.   
    
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (boolean): caching for efficiency
            :class:`torch_geometric.nn.conv.digcn_conv`.
            (default: :obj:`None`)    
    """
    def __init__(self, in_channels, out_channels, cached=True,
                    bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None 
        
    def forward(self, x : Tensor, edge_index : Adj, edge_weight: OptTensor = None):

        x = torch.matmul(x, self.weight)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm
        
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
