import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RootedEgoNets, RootedRWSubgraph

def test_rooted_ego_nets():
    x = torch.randn(3, 1)
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    subgraph_data = RootedEgoNets(2)(data)
    print("")
    print(subgraph_data.subgraph_batch)
    print(subgraph_data.subgraph_nodes_mapper)
    print(subgraph_data.subgraph_edges_mapper)


def test_rooted_rw_subgraph():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)
    DataLoader(RootedRWSubgraph(4)(data))