import torch
from torch_geometric.data import Data, DataLoader

from heat_conv_oct22 import HEATConv

def test_heat_conv():
    ## Prepare a data
    node_features = torch.randn((3, 4))
    node_type = torch.tensor([0, 1, 2])
    edge_type = torch.tensor([0, 2, 1, 1, 2], dtype=torch.long)
    edge_attr = torch.randn((5, 2))
    edge_index = torch.tensor([[0, 0, 1, 2, 0],
                               [1, 2, 0, 0, 0]], dtype=torch.long)
    data_1 = Data(  x=node_features, edge_index=edge_index, 
                    node_type=node_type, edge_type=edge_type,
                    edge_attr=edge_attr )
    data_2 = data_1

    ## Data loader
    dataloader = DataLoader([data_1, data_2], batch_size=2)

    ## Initialize the HEATConv layers accordingly
    heatlayer_1 = HEATConv( in_channels=4, edge_dim=2, 
                            num_node_types = 3, num_edge_types = 3, 
                            edge_attr_emb_size=5, edge_type_emb_size=5, node_emb_size=5, 
                            out_channels=6, heads=3, concat=True)

    heat_2_in_ch = heatlayer_1.out_channels*heatlayer_1.heads if heatlayer_1.concat else heatlayer_1.out_channels
    heatlayer_2 = HEATConv(in_channels=heat_2_in_ch, edge_dim=2, 
                            num_node_types = 3, num_edge_types = 4, 
                            edge_attr_emb_size=5, edge_type_emb_size=5, node_emb_size=5, 
                            out_channels=6, heads=3, concat=False)

    ## Forward
    for d in dataloader:
        x_next = heatlayer_1(d.x, d.edge_index, d.edge_attr, d.node_type, d.edge_type)
        assert x_next.shape == torch.Size([6, 18])
        x_next = heatlayer_2(x_next, d.edge_index, d.edge_attr, d.node_type, d.edge_type)
        assert x_next.shape == torch.Size([6, 6])


if __name__ == '__main__':
    test_heat_conv()
    