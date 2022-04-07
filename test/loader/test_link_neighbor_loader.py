import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


@pytest.mark.parametrize('directed', [True, False])
def test_homogeneous_link_neighbor_loader(directed):
    torch.manual_seed(12345)

    n_edges_pos = 500
    n_edges_neg = 500
    n_nodes = 100
    n_edges = n_edges_pos + n_edges_neg
    batch_size = 10

    positive_edges = get_edge_index(n_nodes, int(n_nodes / 2), n_edges_pos)
    negative_edges = get_edge_index(n_nodes, int(n_nodes / 2), n_edges_neg)
    negative_edges[1, :] += int(n_nodes / 2)
    input_edges = torch.cat([positive_edges, negative_edges], axis=1)
    input_edge_labels = torch.cat([
        torch.Tensor([True] * n_edges_pos),
        torch.Tensor([False] * n_edges_neg)
    ]).type(torch.bool)

    data = Data()
    data.edge_index = positive_edges
    data.x = torch.arange(n_nodes)
    data.edge_attr = torch.arange(n_edges)

    loader = LinkNeighborLoader(data, num_neighbors=[5] * 2,
                                input_edges=input_edges,
                                input_edge_labels=input_edge_labels,
                                batch_size=batch_size, directed=directed,
                                shuffle=True)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == n_edges / batch_size

    for batch in loader:

        if batch.edge_index.size()[1] == 0:
            assert torch.allclose(batch.sampled_edge_labels, False)
            continue

        assert isinstance(batch, Data)

        # assert batch size is correct
        assert batch.batch_size == batch_size

        # assert no additional edges
        assert batch.x.min() >= 0 and batch.x.max() < n_nodes

        # assert edge index makes sense
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes

        # assert also true for sampled edges
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes

        # assert positive samples were present in original graph
        original_edge_set = unique_edge_pairs(data.edge_index)
        batch_edge_index_positive = batch.sampled_edges[:, batch.
                                                        sampled_edge_labels]
        batch_edge = torch.stack([
            batch.x[batch_edge_index_positive[0]],
            batch.x[batch_edge_index_positive[1]]
        ])
        batch_edge_set = unique_edge_pairs(batch_edge)
        assert batch_edge_set <= original_edge_set

        # assert negative samples were not present in original graph
        batch_edge_index_negative = batch.sampled_edges[:, ~batch.
                                                        sampled_edge_labels]
        batch_edge = torch.stack([
            batch.x[batch_edge_index_negative[0]],
            batch.x[batch_edge_index_negative[1]]
        ])
        batch_edge_set = unique_edge_pairs(batch_edge)
        assert len(batch_edge_set.intersection(original_edge_set)) == 0


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_neighbor_loader(directed):
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
