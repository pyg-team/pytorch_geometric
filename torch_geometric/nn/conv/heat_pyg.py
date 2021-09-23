import torch
import torch.nn as nn

def retrieve_masks(node_or_edge_type, wanted_types=[]):
    assert set(wanted_types) < set(node_or_edge_type)
    return [True if x in wanted_types else False for x in node_or_edge_type]

def flatten_a_list_of_list(list_of_list):
    flat_list = []
    for l in list_of_list:
        flat_list += l
    return flat_list

class HEATConv(nn.Module):
    ''' 
        1.  type-specific transformation for nodes of different types: ('n1', 'n2').
            transform nodes from different vector space to the same vector space.
        2.  edges are assumed to have different types but contains the same attributes.            
    '''
    def __init__(self,  node_type_set = ('n1', 'n2', 'n3'), edge_type_set = ('e1', 'e2', 'e3', 'e4'),
                        in_channels_node=4, in_channels_edge_attr=2, in_channels_edge_type=2, 
                        edge_attr_emb_size=5, edge_type_emb_size=5, node_emb_size=5, 
                        out_channels=6, heads=3, concat=True):

        super(HEATlayer, self).__init__()
        ## Parameters
        self.node_type_set = node_type_set
        self.edge_type_set = edge_type_set
        self.in_channels_node = in_channels_node # 32
        self.in_channels_edge_attr = in_channels_edge_attr # 2
        self.in_channels_edge_type = in_channels_edge_type # 2
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_attr_emb_size = edge_attr_emb_size
        self.edge_type_emb_size = edge_type_emb_size
        self.node_emb_size = node_emb_size

        #### Layers ####
        ## Embeddings 
        self.set_node_emb()
        self.set_edge_emb()

        ## Transform the concatenated edge_nbrs feature to out_channels to update the next node feature
        self.node_update_emb = nn.Linear(self.edge_attr_emb_size + self.node_emb_size, self.out_channels, bias=False) 

        ## Attention
        self.attention_nn = nn.Linear(self.edge_attr_emb_size + self.edge_type_emb_size + 2*self.node_emb_size, 1*self.heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.soft_max = nn.Softmax(dim=1)
    
    def set_node_emb(self):
        ''' assume that different nodes have the same dimension, but different vector space. '''
        for node_type in self.node_type_set:
            exec('self.{}_node_feat_emb = nn.Linear(self.in_channels_node, self.node_emb_size, bias=False)'.format(node_type))
    
    def set_edge_emb(self):
        ''' assume that different edges have different types but contains the same kind of attributes. '''
        # for edge_type in self.edge_type_set:
        #     exec('self.{}_edge_type_emb = nn.Linear(self.in_channels_edge_type, self.edge_type_emb_size, bias=False) '.format(edge_type))
        self.edge_type_emb = nn.Linear(self.in_channels_edge_type, self.edge_type_emb_size, bias=False) 
        self.edge_attr_emb = nn.Linear(self.in_channels_edge_attr, self.edge_attr_emb_size, bias=False) 

    def embed_nodes(self, node_features, node_type_list):
        ''' embed node attributes according to their types. '''
        emb_node_features = torch.zeros(node_features.shape[0], self.node_emb_size).to(node_features.device)
        for node_type in self.node_type_set:
            node_mask = retrieve_masks(node_type_list, wanted_types=[node_type])
            exec('emb_node_features[node_mask] = self.{}_node_feat_emb(node_features[node_mask])'.format(node_type))
        return emb_node_features
    
    def embed_edges(self, edge_attrs, edge_types):
        ''' embed edge attributes and edge types respectively. '''
        emb_edge_attributes = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        emb_edge_types = self.leaky_relu(self.edge_type_emb(edge_types))
        return emb_edge_attributes, emb_edge_types
    
    def forward(self, node_feats, edge_index, edge_attr, edge_type, node_type_list, edge_type_list):
        """
        Args:
            node_f:[num_node, in_channels_node]
            edge_index: [2, number_edge]
            edge_attr: [number_edge, len_edge_attr]
            edge_type: [number_edge, len_edge_type]
            node_type_list: [ type of each node ]
            edge_type_list: [ type of each edge ]
        """
        # Node-type specific transformation
        node_emb = self.embed_nodes(node_feats, node_type_list)

        # Edge attribute and type transformation
        edge_attr_emb, edge_type_emb = self.embed_edges(edge_attr, edge_type)

        # Edge-enhanced masked attention
        emb_edge_feat = torch.cat((edge_attr_emb, edge_type_emb), dim=1) # e_{i,j} in heat paper

        src_node_emb = node_emb[edge_index[0]]
        tar_node_emb = node_emb[edge_index[1]] 
        src_node_n_edge_feat = torch.cat((emb_edge_feat, src_node_emb), dim=1) # e^+_{ij}

        ## ------------------------ v Eq. 9 v ------------------------ ## 
        tar_src_node_n_edge_feat = torch.cat((tar_node_emb, src_node_n_edge_feat), dim=1) # eq.9 [ \arrow{h}_{\kappai} | e^+_{i,j}]
        scores = self.leaky_relu(self.attention_nn(tar_src_node_n_edge_feat))
        scores_matrix = torch.sparse_coo_tensor(edge_index, scores, torch.Size([node_feats.shape[0], node_feats.shape[0], self.heads])).to_dense().to(node_feats.device)
        scores_matrix = torch.where(scores_matrix==0.0, torch.ones_like(scores_matrix) * -10000, scores_matrix)
        attention_matrix = self.soft_max(scores_matrix) # [num_nodes, num_nbrs, heads]]
        ## =========================================================== ## 

        ## ------------------------ v Eq. 10 v ------------------------ ## 
        src_n_edge_attr = self.leaky_relu(self.node_update_emb(torch.cat((src_node_emb, edge_attr_emb), dim=1))) # eq.10 W_h[ concat. ]
        src_n_edge_attr_matrix = torch.sparse_coo_tensor(edge_index, src_n_edge_attr, 
                                    torch.Size([node_feats.shape[0], node_feats.shape[0], self.out_channels])).to_dense().to(node_feats.device)
        
        attention_matrix = attention_matrix.unsqueeze(3).repeat(1,1,1,self.out_channels)
        src_n_edge_attr_matrix = src_n_edge_attr_matrix.unsqueeze(dim=2).repeat(1, 1, self.heads, 1)

        next_node_feat = torch.sum(torch.mul(attention_matrix,  src_n_edge_attr_matrix), dim=1)
        ## =========================================================== ## 
        if self.concat:
            next_node_feat = torch.flatten(next_node_feat, start_dim=1)
        else:
            next_node_feat = torch.mean(next_node_feat, dim=1)
        return next_node_feat

if __name__ == '__main__':
    from torch_geometric.data import Data, DataLoader

    ## Pre-define the node_type_set and edge_type_set to cover all the node and edge types in the dataset.
    node_type_set = ('n1', 'n2', 'n3')
    edge_type_set = ('e1', 'e2', 'e3', 'e4')

    ## Prepare a data
    node_features = torch.randn((3, 4))
    node_type_set = ('n1', 'n2', )
    edge_type_set = ('e1', 'e2', 'e3')

    node_type_list = ['n1', 'n2', 'n2']
    edge_type_list = ['e1', 'e3', 'e2', 'e2', 'e3']
    edge_type = torch.randn((5, 2))
    edge_attr = torch.randn((5, 2))
    edge_index = torch.tensor([[0, 0, 1, 2, 0],
                               [1, 2, 0, 0, 0]])
    data_1 = Data(  x=node_features, edge_index=edge_index, 
                    node_type_list=node_type_list, edge_type_list=edge_type_list,
                    edge_type=edge_type, edge_attr=edge_attr )
                    
    node_features = torch.randn((3, 4))
    

    node_type_list = ['n1', 'n3', 'n2']
    edge_type_list = ['e1', 'e3', 'e4', 'e2', 'e3']
    edge_type = torch.randn((5, 2))
    edge_attr = torch.randn((5, 2))
    edge_index = torch.tensor([[0, 0, 1, 2, 0],
                               [1, 2, 0, 0, 0]])
    data_2 = Data(  x=node_features, edge_index=edge_index, 
                    node_type_list=node_type_list, edge_type_list=edge_type_list,
                    edge_type=edge_type, edge_attr=edge_attr )

    dataloader = DataLoader([data_1, data_2], batch_size=1)

    ## initialize the HEAT layers accordingly, node_type_set and edge_type_set
    heatlayer_1 = HEATlayer(in_channels_node=4, node_type_set=node_type_set, edge_type_set=edge_type_set)
    heatlayer_2 = HEATlayer(in_channels_node=6*heatlayer_1.heads, node_type_set=node_type_set, edge_type_set=edge_type_set)
    heatlayer_3 = HEATlayer(in_channels_node=6*heatlayer_2.heads, node_type_set=node_type_set, edge_type_set=edge_type_set)


    for d in dataloader:
        print(d)
        d.flat_node_type_list = flatten_a_list_of_list(d.node_type_list)
        d.flat_edge_type_list = flatten_a_list_of_list(d.edge_type_list)
        print(d)

        x_next = heatlayer_1(d.x, d.edge_index, d.edge_attr, d.edge_type, d.flat_node_type_list, d.flat_edge_type_list)
        x_next = heatlayer_2(x_next, d.edge_index, d.edge_attr, d.edge_type, d.flat_node_type_list, d.flat_edge_type_list)
        x_next = heatlayer_3(x_next, d.edge_index, d.edge_attr, d.edge_type, d.flat_node_type_list, d.flat_edge_type_list)
        print(x_next.shape)
        # break





    


    