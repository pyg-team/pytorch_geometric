import torch
from torch import index_select
from torch.nn import Linear, Tanh, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, segment_csr #already required by MessagePassing

try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None
    

class GravNetConv(MessagePassing):
    r"""The GravNet from the `"Learning representations of irregular
    particle-detector geometry with distance-weighted graph
    networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors. The neighbors are constructed
    in a learnable low-dimensional projection of the feature space.
    A second projection of the input feature space is then propagated 
    from the neighbors to each vertex using distance weights
    that are derived by applying a Gaussian function to the distances.
    
    Args:
        in_channels (int): The number of input channels.
        space_dimensions_s (int): The dimensionality of the space
           used to construct the neighbors; referred to as "S" in the paper.
        propagate_dimensions_flr (int): The number of features to be propagated
           between the vertices; referred to as "F_LR" in the paper.
        k_neighbors (int): The number of nearest neighbors.
        out_channels_fout (int): The number of output channels; referred to as
          "F_OUT" in the paper.
    """
    def __init__(self, 
                 in_channels, 
                 space_dimensions_s,
                 propagate_dimensions_flr,
                 k_neighbors,
                 out_channels_fout):
        super(GravNetConv, self).__init__(aggr='max') # aggr='max' will be overwritten
        
        if knn_graph is None:
            raise ImportError('`GravNetConv` requires `torch-cluster`.')
        
        self.dense_s    = Linear(in_channels, space_dimensions_s)
        self.dense_flr  = Linear(in_channels, propagate_dimensions_flr)
        self.dense_fout = Sequential(Linear(in_channels+2*propagate_dimensions_flr, out_channels_fout), Tanh())
        
        self.k = k_neighbors

    def forward(self, x, batch=None):

        spatial = self.dense_s(x)
        to_propagate = self.dense_flr(x)
        
        edge_index = knn_graph(spatial, self.k, batch, loop=False, flow=self.flow)
        
        reference = index_select(spatial,0,edge_index[1])
        neighbors = index_select(spatial,0,edge_index[0])
        
        distancessq = torch.sum((reference-neighbors)**2, dim=-1)
        distance_weights = torch.exp(-10.*distancessq) #factor 10 gives a better initial spread
                
        prop_feat = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=to_propagate, edge_weights=distance_weights)
        
        return self.dense_fout(torch.cat([prop_feat,x],dim=1))

    #overwrite for mean and max
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        
        #this is probably not needed
        if ptr is not None:
            raise ValueError('`GravNetConv` does not support `ptr` in aggregate.')
        return torch.cat([scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,reduce="mean"),
                       scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,reduce="max")], dim=1)
     
    def message(self, x_i, x_j, edge_weights):
        return x_j*edge_weights.unsqueeze(1)
        
    def update(self, aggr_out):
        return aggr_out
    
    
    
    
