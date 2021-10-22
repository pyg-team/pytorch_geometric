import torch
from torch import Tensor, LongTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear

class HEATConv(MessagePassing):
    '''
    1. type-specific transformation for nodes of different types.
       transform nodes from different vector space to the same vector space.
    2. edges are assumed to have different types but contains the same kind of attributes.
    '''
    def __init__(self,  in_channels: int, edge_dim: int, 
                        num_node_types: int, num_edge_types: int,
                        edge_attr_emb_size: int, edge_type_emb_size: int, node_emb_size: int, 
                        out_channels: int, heads: int, concat: bool = True, **kwargs):

        super().__init__(aggr='add', **kwargs)
        ## Parameters
        self.in_channels = in_channels 
        self.edge_dim = edge_dim 

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
    
        self.edge_attr_emb_size = edge_attr_emb_size
        self.edge_type_emb_size = edge_type_emb_size
        self.node_emb_size = node_emb_size
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        #### Layers ####
        ## Node Embeddings 
        ''' assuming that different nodes have the same dimension, but different vector space. '''
        self.het_node_lin = HeteroLinear(in_channels=self.in_channels, 
                                             num_node_types=self.num_node_types, 
                                             node_emb_size=self.node_emb_size)
        ## Edge Embeddings
        ''' assuming that different edges have different types but contains the same kind of attributes. '''
        self.edge_type_emb = torch.nn.Embedding(self.num_edge_types, self.edge_type_emb_size)
        self.edge_attr_emb = torch.nn.Linear(self.edge_dim, self.edge_attr_emb_size, bias=False) 

        ## Transform the concatenated edge_nbrs feature to out_channels to update the next node feature
        self.node_update_emb = torch.nn.Linear(self.edge_attr_emb_size + self.node_emb_size, self.out_channels, bias=False) 

        ## Attention
        self.attention_nn = torch.nn.Linear(self.edge_attr_emb_size + self.edge_type_emb_size + 2*self.node_emb_size, 1*self.heads, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.soft_max = torch.nn.Softmax(dim=1)
    
    
    def embed_edges(self, edge_attrs, edge_types):
        ''' embed edge attributes and edge types respectively. '''
        emb_edge_attributes = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        emb_edge_types = self.leaky_relu(self.edge_type_emb(edge_types))
        return emb_edge_attributes, emb_edge_types
  

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, node_type: LongTensor, edge_type: LongTensor):
        """
        Args:
            x:[num_node, in_channels]
            edge_index: [2, number_edge]
            edge_attr: [number_edges, len_edge_attr]
            node_type: [number_nodes, 1] node_type is an integer
            edge_type: [number_edges, 1] edge_type is an integer
        """
        # Node-type specific transformation
        node_emb = self.het_node_lin(x, node_type)

        # Edge attribute and type transformation
        edge_attr_emb, edge_type_emb = self.embed_edges(edge_attr, edge_type)
        
        # Edge-enhanced masked attention
        emb_edge_feat = torch.cat((edge_attr_emb, edge_type_emb), dim=1) # e_{i,j} in heat paper

        src_node_emb = node_emb[edge_index[0]]
        tar_node_emb = node_emb[edge_index[1]] 
        src_node_n_edge_feat = torch.cat((emb_edge_feat, src_node_emb), dim=1) # e^+_{ij}

        ## ------------------------ v Eq. 9 v ------------------------ ## 
        tar_src_node_n_edge_feat = torch.cat((tar_node_emb, src_node_n_edge_feat), dim=1) # eq.9 [ \arrow{h}_{\kappa i} | e^+_{i,j}]
        scores = self.leaky_relu(self.attention_nn(tar_src_node_n_edge_feat))
        
        scores_matrix = torch.sparse_coo_tensor(edge_index, scores, torch.Size([x.shape[0], x.shape[0], self.heads])).to_dense().to(x.device)
        scores_matrix = torch.where(scores_matrix==0.0, torch.ones_like(scores_matrix) * -10000, scores_matrix)
        attention_matrix = self.soft_max(scores_matrix) # [num_nodes, num_nbrs, heads]]

        edges_attentions_index = torch.nonzero(attention_matrix, as_tuple=True) # indexes in the att matrix
        edges_attentions = attention_matrix[edges_attentions_index]
        ## ============================================================ ## 

        ## ------------------------ v Eq. 10 v ------------------------ ## 
        src_n_edge_attr = self.leaky_relu(self.node_update_emb(torch.cat((src_node_emb, edge_attr_emb), dim=1))) # eq.10 W_h[ concat. ]
        ## =========================================================== ## 

        return self.propagate(edge_index, x=src_n_edge_attr, norm=edges_attentions)
        

    ## ------------------------ Message Passing ------------------------ ## 
    def message(self, x, norm):
        x = x.unsqueeze(2).repeat(1,1,self.heads)
        norm = norm.view(x.shape[0], 1, self.heads).repeat(1, self.out_channels, 1)
        
        return torch.flatten(x*norm, start_dim=1) if self.concat else torch.mean(x*norm, dim=2)
        





    


    
