import pytest
import torch

from torch_geometric.data import Data, HeteroData


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.fixture
def hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)

    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    data['paper', 'author'].edge_attr = torch.randn(10, 8)

    return data


@pytest.fixture
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ]),
        edge_attr=torch.randn(6, 3),
    )
