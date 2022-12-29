from torch_geometric.loader import DataLoader


def test_citeseer(get_dataset):
    dataset = get_dataset(name='CiteSeer')
    loader = DataLoader(dataset, batch_size=len(dataset))

    assert len(dataset) == 1
    assert dataset.__repr__() == 'CiteSeer()'

    for batch in loader:
        assert batch.num_graphs == len(batch) == 1
        assert batch.num_nodes == 3327
        assert batch.num_edges / 2 == 4552

        assert list(batch.x.size()) == [batch.num_nodes, 3703]
        assert list(batch.y.size()) == [batch.num_nodes]
        assert batch.y.max() + 1 == 6
        assert batch.train_mask.sum() == 6 * 20
        assert batch.val_mask.sum() == 500
        assert batch.test_mask.sum() == 1000
        assert (batch.train_mask & batch.val_mask & batch.test_mask).sum() == 0
        assert list(batch.batch.size()) == [batch.num_nodes]
        assert batch.ptr.tolist() == [0, batch.num_nodes]

        assert batch.has_isolated_nodes()
        assert not batch.has_self_loops()
        assert batch.is_undirected()


def test_citeseer_with_full_split(get_dataset):
    dataset = get_dataset(name='CiteSeer', split='full')
    data = dataset[0]
    assert data.val_mask.sum() == 500
    assert data.test_mask.sum() == 1000
    assert data.train_mask.sum() == data.num_nodes - 1500
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0


def test_citeseer_with_random_split(get_dataset):
    dataset = get_dataset(name='CiteSeer', split='random',
                          num_train_per_class=11, num_val=29, num_test=41)
    data = dataset[0]
    assert data.train_mask.sum() == dataset.num_classes * 11
    assert data.val_mask.sum() == 29
    assert data.test_mask.sum() == 41
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0
