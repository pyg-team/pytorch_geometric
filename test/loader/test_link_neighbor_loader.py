import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize('directed', [True, False])
def test_homogeneous_link_neighbor_loader(directed):
    torch.manual_seed(12345)

    pos_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index[1, :] += 50

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(500), torch.zeros(500)], dim=0)

    data = Data()

    data.edge_index = pos_edge_index
    data.x = torch.arange(100)
    data.edge_attr = torch.arange(500)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2, batch_size=20,
                                edge_label_index=edge_label_index,
                                edge_label=edge_label, directed=directed,
                                shuffle=True)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, Data)

        assert len(batch) == 5
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        # Assert positive samples were present in the original graph:
        edge_index = unique_edge_pairs(batch.edge_index)
        edge_label_index = batch.edge_label_index[:, batch.edge_label == 1]
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index | edge_label_index) == len(edge_index)

        # Assert negative samples were not present in the original graph:
        edge_index = unique_edge_pairs(batch.edge_index)
        edge_label_index = batch.edge_label_index[:, batch.edge_label == 0]
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index & edge_label_index) == 0


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_neighbor_loader(directed):
    return
    torch.manual_seed(12345)

    batch_size = 20
    n_paper = 100
    n_author = 200
    n_paper_paper = 500
    n_paper_author = 1000
    n_author_paper = 500

    data = HeteroData()

    data['paper'].x = torch.arange(n_paper)
    data['author'].x = torch.arange(n_author)

    data['paper', 'paper'].edge_index = get_edge_index(n_paper, n_paper,
                                                       n_paper_paper)
    data['paper', 'paper'].edge_attr = torch.arange(n_paper_paper)
    data['paper',
         'author'].edge_index = get_edge_index(n_paper, n_author,
                                               n_paper_author)
    data['paper', 'author'].edge_attr = torch.arange(n_paper_author)
    data['author',
         'paper'].edge_index = get_edge_index(n_author, n_paper,
                                              n_author_paper)
    data['author', 'paper'].edge_attr = torch.arange(n_author_paper)

    loader = LinkNeighborLoader(data, num_neighbors=[10] * 2,
                                input_edges=('paper', 'to', 'author'),
                                batch_size=batch_size, directed=directed)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == int(n_paper_author / batch_size)

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}
