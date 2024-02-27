import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.testing import get_random_edge_index
from torch_geometric.transforms import ChronologicalSplit


@pytest.mark.parametrize('val_ratio', [0.15, 0.2])
@pytest.mark.parametrize('test_ratio', [0.15, 0.2])
def test_chronological_split_on_data(val_ratio, test_ratio):
    num_nodes = 32
    num_edges = 100
    data = Data(x=torch.randn(num_nodes, 12),
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                time=torch.randint(0, 100, (num_edges, )), num_nodes=num_nodes)
    transform = ChronologicalSplit(val_ratio=val_ratio, test_ratio=test_ratio)
    train_data, val_data, test_data = transform(data)

    assert train_data.time.max() < val_data.time.min()
    assert val_data.time.max() < test_data.time.min()


@pytest.mark.parametrize('val_ratio', [0.15, 0.2])
@pytest.mark.parametrize('test_ratio', [0.15, 0.2])
def test_chronological_split_on_hetero_data(val_ratio, test_ratio):
    num_paper_nodes = 32
    num_author_nodes = 64
    num_edges = 100
    data = HeteroData()
    data['paper'].x = torch.randn(num_paper_nodes, 16)
    data['paper'].time = torch.randint(0, 100, (num_paper_nodes, ))
    data['author'].x = torch.randn(num_author_nodes, 16)
    data['author'].time = torch.randint(100, 200, (num_author_nodes, ))
    data['paper',
         'author'].edge_index = get_random_edge_index(num_paper_nodes,
                                                      num_author_nodes,
                                                      num_edges)
    data['paper', 'author'].time = torch.randint(200, 300, (num_edges, ))
    transform = ChronologicalSplit(val_ratio=val_ratio, test_ratio=test_ratio)
    train_data, val_data, test_data = transform(data)

    for key in ['paper', 'author', ('paper', 'author')]:
        assert train_data[key].time.max() < val_data[key].time.min()
        assert val_data[key].time.max() < test_data[key].time.min()
