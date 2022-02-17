import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomNodeSplit


@pytest.mark.parametrize('num_splits', [1, 2])
def test_random_node_split(num_splits):
    num_nodes, num_classes = 1000, 4
    x = torch.randn(num_nodes, 16)
    y = torch.randint(num_classes, (num_nodes, ), dtype=torch.long)
    data = Data(x=x, y=y)

    transform = RandomNodeSplit(split='train_rest', num_splits=num_splits,
                                num_val=100, num_test=200)
    assert str(transform) == 'RandomNodeSplit(split=train_rest)'
    data = transform(data)
    assert len(data) == 5

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    assert train_mask.size() == (num_nodes, num_splits)
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    assert val_mask.size() == (num_nodes, num_splits)
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask
    assert test_mask.size() == (num_nodes, num_splits)

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == num_nodes - 100 - 200
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == 200
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == num_nodes)

    transform = RandomNodeSplit(split='train_rest', num_splits=num_splits,
                                num_val=0.1, num_test=0.2)
    data = transform(data)

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == num_nodes - 100 - 200
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == 200
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == num_nodes)

    transform = RandomNodeSplit(split='test_rest', num_splits=num_splits,
                                num_train_per_class=10, num_val=100)
    assert str(transform) == 'RandomNodeSplit(split=test_rest)'
    data = transform(data)
    assert len(data) == 5

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == 10 * num_classes
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == num_nodes - 10 * num_classes - 100
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == num_nodes)

    transform = RandomNodeSplit(split='test_rest', num_splits=num_splits,
                                num_train_per_class=10, num_val=0.1)
    data = transform(data)

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == 10 * num_classes
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == num_nodes - 10 * num_classes - 100
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == num_nodes)

    transform = RandomNodeSplit(split='random', num_splits=num_splits,
                                num_train_per_class=10, num_val=100,
                                num_test=200)
    assert str(transform) == 'RandomNodeSplit(split=random)'
    data = transform(data)
    assert len(data) == 5

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == 10 * num_classes
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == 200
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == 10 * num_classes + 100 + 200)

    transform = RandomNodeSplit(split='random', num_splits=num_splits,
                                num_train_per_class=10, num_val=0.1,
                                num_test=0.2)
    assert str(transform) == 'RandomNodeSplit(split=random)'
    data = transform(data)

    train_mask = data.train_mask
    train_mask = train_mask.unsqueeze(-1) if num_splits == 1 else train_mask
    val_mask = data.val_mask
    val_mask = val_mask.unsqueeze(-1) if num_splits == 1 else val_mask
    test_mask = data.test_mask
    test_mask = test_mask.unsqueeze(-1) if num_splits == 1 else test_mask

    for i in range(train_mask.size(-1)):
        assert train_mask[:, i].sum() == 10 * num_classes
        assert val_mask[:, i].sum() == 100
        assert test_mask[:, i].sum() == 200
        assert (train_mask[:, i] & val_mask[:, i] & test_mask[:, i]).sum() == 0
        assert ((train_mask[:, i] | val_mask[:, i]
                 | test_mask[:, i]).sum() == 10 * num_classes + 100 + 200)


def test_random_node_split_on_hetero_data():
    data = HeteroData()

    data['paper'].x = torch.randn(2000, 16)
    data['paper'].y = torch.randint(4, (2000, ), dtype=torch.long)
    data['author'].x = torch.randn(300, 16)

    transform = RandomNodeSplit()
    assert str(transform) == 'RandomNodeSplit(split=train_rest)'
    data = transform(data)
    assert len(data) == 5

    assert len(data['author']) == 1
    assert len(data['paper']) == 5

    assert data['paper'].train_mask.sum() == 500
    assert data['paper'].val_mask.sum() == 500
    assert data['paper'].test_mask.sum() == 1000
