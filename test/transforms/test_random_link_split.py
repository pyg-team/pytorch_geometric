import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.testing import get_random_edge_index
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import is_undirected, to_undirected


def test_random_link_split():
    assert str(RandomLinkSplit()) == ('RandomLinkSplit('
                                      'num_val=0.1, num_test=0.2)')

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    edge_attr = torch.randn(edge_index.size(1), 3)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=100)

    # No test split:
    transform = RandomLinkSplit(num_val=2, num_test=0, is_undirected=True)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert train_data.edge_index.size() == (2, 6)
    assert train_data.edge_attr.size() == (6, 3)
    assert train_data.edge_label_index.size(1) == 6
    assert train_data.edge_label.size(0) == 6

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert val_data.edge_index.size() == (2, 6)
    assert val_data.edge_attr.size() == (6, 3)
    assert val_data.edge_label_index.size(1) == 4
    assert val_data.edge_label.size(0) == 4

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert test_data.edge_index.size() == (2, 10)
    assert test_data.edge_attr.size() == (10, 3)
    assert test_data.edge_label_index.size() == (2, 0)
    assert test_data.edge_label.size() == (0, )

    # Percentage split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                neg_sampling_ratio=2.0, is_undirected=False)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert train_data.edge_index.size() == (2, 6)
    assert train_data.edge_attr.size() == (6, 3)
    assert train_data.edge_label_index.size(1) == 18
    assert train_data.edge_label.size(0) == 18

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert val_data.edge_index.size() == (2, 6)
    assert val_data.edge_attr.size() == (6, 3)
    assert val_data.edge_label_index.size(1) == 6
    assert val_data.edge_label.size(0) == 6

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert test_data.edge_index.size() == (2, 8)
    assert test_data.edge_attr.size() == (8, 3)
    assert test_data.edge_label_index.size(1) == 6
    assert test_data.edge_label.size(0) == 6

    # Disjoint training split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=False,
                                disjoint_train_ratio=0.5)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert train_data.edge_index.size() == (2, 3)
    assert train_data.edge_attr.size() == (3, 3)
    assert train_data.edge_label_index.size(1) == 6
    assert train_data.edge_label.size(0) == 6


def test_random_link_split_with_label():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    edge_label = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    data = Data(edge_index=edge_index, edge_label=edge_label, num_nodes=6)

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                neg_sampling_ratio=0.0)
    train_data, _, _ = transform(data)
    assert len(train_data) == 4
    assert train_data.num_nodes == 6
    assert train_data.edge_index.size() == (2, 6)
    assert train_data.edge_label_index.size() == (2, 6)
    assert train_data.edge_label.size() == (6, )
    assert train_data.edge_label.min() == 0
    assert train_data.edge_label.max() == 1

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                neg_sampling_ratio=1.0)
    train_data, _, _ = transform(data)
    assert len(train_data) == 4
    assert train_data.num_nodes == 6
    assert train_data.edge_index.size() == (2, 6)
    assert train_data.edge_label_index.size() == (2, 12)
    assert train_data.edge_label.size() == (12, )
    assert train_data.edge_label.min() == 0
    assert train_data.edge_label.max() == 2
    assert train_data.edge_label[6:].sum() == 0


def test_random_link_split_increment_label():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    edge_label = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    data = Data(edge_index=edge_index, edge_label=edge_label, num_nodes=6)

    transform = RandomLinkSplit(num_val=0, num_test=0, neg_sampling_ratio=0.0)
    train_data, _, _ = transform(data)
    assert train_data.edge_label.numel() == edge_index.size(1)
    assert train_data.edge_label.min() == 0
    assert train_data.edge_label.max() == 1

    transform = RandomLinkSplit(num_val=0, num_test=0, neg_sampling_ratio=1.0)
    train_data, _, _ = transform(data)
    assert train_data.edge_label.numel() == 2 * edge_index.size(1)
    assert train_data.edge_label.min() == 0
    assert train_data.edge_label.max() == 2
    assert train_data.edge_label[edge_index.size(1):].sum() == 0


def test_random_link_split_on_hetero_data():
    data = HeteroData()

    data['p'].x = torch.arange(100)
    data['a'].x = torch.arange(100, 300)

    data['p', 'p'].edge_index = get_random_edge_index(100, 100, 500)
    data['p', 'p'].edge_index = to_undirected(data['p', 'p'].edge_index)
    data['p', 'p'].edge_attr = torch.arange(data['p', 'p'].num_edges)
    data['p', 'a'].edge_index = get_random_edge_index(100, 200, 1000)
    data['p', 'a'].edge_attr = torch.arange(500, 1500)
    data['a', 'p'].edge_index = data['p', 'a'].edge_index.flip([0])
    data['a', 'p'].edge_attr = torch.arange(1500, 2500)

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True,
                                edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 4
    assert len(train_data['p', 'a']) == 2
    assert len(train_data['a', 'p']) == 2

    assert is_undirected(train_data['p', 'p'].edge_index,
                         train_data['p', 'p'].edge_attr)
    assert is_undirected(val_data['p', 'p'].edge_index,
                         val_data['p', 'p'].edge_attr)
    assert is_undirected(test_data['p', 'p'].edge_index,
                         test_data['p', 'p'].edge_attr)

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                edge_types=('p', 'a'),
                                rev_edge_types=('a', 'p'))
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 2
    assert len(train_data['p', 'a']) == 4
    assert len(train_data['a', 'p']) == 2

    assert train_data['p', 'a'].edge_index.size() == (2, 600)
    assert train_data['p', 'a'].edge_attr.size() == (600, )
    assert train_data['p', 'a'].edge_attr.min() >= 500
    assert train_data['p', 'a'].edge_attr.max() <= 1500
    assert train_data['a', 'p'].edge_index.size() == (2, 600)
    assert train_data['a', 'p'].edge_attr.size() == (600, )
    assert train_data['a', 'p'].edge_attr.min() >= 500
    assert train_data['a', 'p'].edge_attr.max() <= 1500
    assert train_data['p', 'a'].edge_label_index.size() == (2, 1200)
    assert train_data['p', 'a'].edge_label.size() == (1200, )

    assert val_data['p', 'a'].edge_index.size() == (2, 600)
    assert val_data['p', 'a'].edge_attr.size() == (600, )
    assert val_data['p', 'a'].edge_attr.min() >= 500
    assert val_data['p', 'a'].edge_attr.max() <= 1500
    assert val_data['a', 'p'].edge_index.size() == (2, 600)
    assert val_data['a', 'p'].edge_attr.size() == (600, )
    assert val_data['a', 'p'].edge_attr.min() >= 500
    assert val_data['a', 'p'].edge_attr.max() <= 1500
    assert val_data['p', 'a'].edge_label_index.size() == (2, 400)
    assert val_data['p', 'a'].edge_label.size() == (400, )

    assert test_data['p', 'a'].edge_index.size() == (2, 800)
    assert test_data['p', 'a'].edge_attr.size() == (800, )
    assert test_data['p', 'a'].edge_attr.min() >= 500
    assert test_data['p', 'a'].edge_attr.max() <= 1500
    assert test_data['a', 'p'].edge_index.size() == (2, 800)
    assert test_data['a', 'p'].edge_attr.size() == (800, )
    assert test_data['a', 'p'].edge_attr.min() >= 500
    assert test_data['a', 'p'].edge_attr.max() <= 1500
    assert test_data['p', 'a'].edge_label_index.size() == (2, 400)
    assert test_data['p', 'a'].edge_label.size() == (400, )

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True,
                                edge_types=[('p', 'p'), ('p', 'a')],
                                rev_edge_types=[None, ('a', 'p')])
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 4
    assert len(train_data['p', 'a']) == 4
    assert len(train_data['a', 'p']) == 2

    assert is_undirected(train_data['p', 'p'].edge_index,
                         train_data['p', 'p'].edge_attr)
    assert train_data['p', 'a'].edge_index.size() == (2, 600)
    assert train_data['a', 'p'].edge_index.size() == (2, 600)

    # No reverse edge types specified:
    transform = RandomLinkSplit(edge_types=[('p', 'p'), ('p', 'a')])
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].num_edges < data['p', 'p'].num_edges
    assert train_data['p', 'a'].num_edges < data['p', 'a'].num_edges
    assert train_data['a', 'p'].num_edges == data['a', 'p'].num_edges


def test_random_link_split_on_undirected_hetero_data():
    data = HeteroData()
    data['p'].x = torch.arange(100)
    data['p', 'p'].edge_index = get_random_edge_index(100, 100, 500)
    data['p', 'p'].edge_index = to_undirected(data['p', 'p'].edge_index)

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'),
                                rev_edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'),
                                rev_edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()


def test_random_link_split_insufficient_negative_edges():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 3, 0, 2, 0, 1]])
    data = Data(edge_index=edge_index, num_nodes=4)

    transform = RandomLinkSplit(num_val=0.34, num_test=0.34,
                                is_undirected=False, neg_sampling_ratio=2,
                                split_labels=True)

    with pytest.warns(UserWarning, match="not enough negative edges"):
        train_data, val_data, test_data = transform(data)

    assert train_data.neg_edge_label_index.size() == (2, 2)
    assert val_data.neg_edge_label_index.size() == (2, 2)
    assert test_data.neg_edge_label_index.size() == (2, 2)
