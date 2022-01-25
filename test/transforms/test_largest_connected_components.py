import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents

def test_largest_connected_components():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 9],
                               [1, 2, 0, 2, 0, 1, 3, 2, 4, 3, 6, 7, 9, 8]])
    data = Data(edge_index=edge_index, num_nodes=10)
    
    lcc = LargestConnectedComponents(num_components=1)

    before_transformation = data.clone()
    data = lcc(data)
    
    assert data.num_nodes == 5 
    
    assert torch.all(data.edge_index == before_transformation.edge_index[:, :10])
    