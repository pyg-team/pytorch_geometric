import torch

def retrieve_masks(node_or_edge_type, wanted_types=[]):
    assert set(wanted_types) < set(node_or_edge_type), 'wanted: {}, contains: {}'.format(set(wanted_types), set(node_or_edge_type))
    return [True if x in wanted_types else False for x in node_or_edge_type]

class HeteroLinear(torch.nn.Module):
    '''
    1. type-specific transformation for nodes of different types: 
    '''
    def __init__(self, in_channels: int, out_channels: int, num_node_types: int, **kwargs):
        super().__init__()

        self.in_channels = in_channels 
        self.num_node_types = num_node_types
        self.node_emb_size = node_emb_size

        self.HeteroNodeEmbedLinears = torch.nn.ModuleList([torch.nn.Linear(self.in_channels, self.node_emb_size) for i in range(self.num_node_types)])
    
    def forward(self, x, node_type_list):
        emb_x = torch.zeros(x.shape[0], self.node_emb_size).to(x.device)
        for node_type in range(self.num_node_types): # node_type is an integer, so that node_type_list should be a list of integers.
            node_mask = retrieve_masks(node_type_list, wanted_types=[node_type])
            emb_x[node_mask] = self.HeteroNodeEmbedLinears[node_type](x[node_mask])
        return emb_x

if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data, DataLoader

    def flatten_a_list_of_list(list_of_list):
        flat_list = []
        for l1 in list_of_list:
            flat_list += l1
        return flat_list

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

    dataloader = DataLoader([data_1, data_2], batch_size=1)

    hetNodeLin = HeteroNodeLinear(in_channels=4, num_node_types=3, node_emb_size=5)

    for d in dataloader:
        # print(d)
        d.flat_node_type_list = flatten_a_list_of_list(d.node_type_list)
        d.flat_edge_type_list = flatten_a_list_of_list(d.edge_type_list)
        
        x_emb = hetNodeLin(d.x, d.flat_node_type_list)
        print(x_emb)

        break

