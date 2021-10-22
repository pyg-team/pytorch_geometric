import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from torch_geometric.nn.dense.linear import HeteroLinear

def flatten_a_list_of_list(list_of_list):
    flat_list = []
    for l1 in list_of_list:
        flat_list += l1
    return flat_list

class HEATConv(MessagePassing):
    '''
    1. type-specific transformation for nodes of different types: ('n1', 'n2').
       transform nodes from different vector space to the same vector space.
    2. edges are assumed to have different types but contains the same attributes.
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
        self.edge_type_emb = nn.Embedding(self.num_edge_types, self.edge_type_emb_size)
        self.edge_attr_emb = nn.Linear(self.edge_dim, self.edge_attr_emb_size, bias=False) 

        ## Transform the concatenated edge_nbrs feature to out_channels to update the next node feature
        self.node_update_emb = nn.Linear(self.edge_attr_emb_size + self.node_emb_size, self.out_channels, bias=False) 

        ## Attention
        self.attention_nn = nn.Linear(self.edge_attr_emb_size + self.edge_type_emb_size + 2*self.node_emb_size, 1*self.heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.soft_max = nn.Softmax(dim=1)
    
    
    def embed_edges(self, edge_attrs, edge_types):
        ''' embed edge attributes and edge types respectively. '''
        emb_edge_attributes = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        emb_edge_types = self.leaky_relu(self.edge_type_emb(torch.tensor(edge_types)))
        return emb_edge_attributes, emb_edge_types
  

    def forward(self, x, edge_index, edge_attr, node_type, edge_type):
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
        
if __name__ == '__main__':
    from torch_geometric.data import Data, DataLoader
    
    ## Prepare a data
    node_features = torch.randn((3, 4))
    
    node_type = [0, 1, 2]
    edge_type = torch.tensor([[0, 2, 1, 1, 2]], dtype=torch.long)
    edge_type = [0, 2, 1, 1, 2]
    edge_attr = torch.randn((5, 2))
    edge_index = torch.tensor([[0, 0, 1, 2, 0],
                               [1, 2, 0, 0, 0]])
    data_1 = Data(  x=node_features, edge_index=edge_index, 
                    node_type_list=node_type, edge_type_list=edge_type,
                    edge_type=edge_type, edge_attr=edge_attr )

    data_2 = data_1

    dataloader = DataLoader([data_1, data_2], batch_size=2)

    ## initialize the HEATConv layers accordingly
    heatlayer_1 = HEATConv(in_channels=4, num_node_types = 3, num_edge_types = 4, heads=3, concat=True)
    heat_2_in_ch = heatlayer_1.out_channels*heatlayer_1.heads if heatlayer_1.concat else heatlayer_1.out_channels
    heatlayer_2 = HEATConv(in_channels=heat_2_in_ch, num_node_types = 3, num_edge_types = 4, concat=False)
    heat_3_in_ch = heatlayer_2.out_channels*heatlayer_2.heads if heatlayer_2.concat else heatlayer_2.out_channels
    heatlayer_3 = HEATConv(in_channels=heat_3_in_ch, num_node_types = 3, num_edge_types = 4, concat=False)
    
    for d in dataloader:
        # print(d)
        d.flat_node_type_list = flatten_a_list_of_list(d.node_type_list)
        d.flat_edge_type_list = flatten_a_list_of_list(d.edge_type_list)
        # print(d)

        x_next = heatlayer_1(d.x, d.edge_index, d.edge_attr, d.flat_node_type_list, d.flat_edge_type_list)
        x_next = heatlayer_2(x_next, d.edge_index, d.edge_attr, d.flat_node_type_list, d.flat_edge_type_list)
        x_next = heatlayer_3(x_next, d.edge_index, d.edge_attr, d.flat_node_type_list, d.flat_edge_type_list)
        print(x_next.shape)
        # break





    


    
