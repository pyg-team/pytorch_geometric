from torch_geometric.loader import DataLoader


def test_citeseer(get_dataset):
    dataset = get_dataset(name='CiteSeer')
    loader = DataLoader(dataset, batch_size=len(dataset))

    assert len(dataset) == 1
    assert dataset.__repr__() == 'CiteSeer()'

    for data in loader:
        assert data.num_graphs == 1
        assert data.num_nodes == 3327
        assert data.num_edges / 2 == 4552

        assert len(data) == 8
        assert list(data.x.size()) == [data.num_nodes, 3703]
        assert list(data.y.size()) == [data.num_nodes]
        assert data.y.max() + 1 == 6
        assert data.train_mask.sum() == 6 * 20
        assert data.val_mask.sum() == 500
        assert data.test_mask.sum() == 1000
        assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0
        assert list(data.batch.size()) == [data.num_nodes]
        assert data.ptr.tolist() == [0, data.num_nodes]

        assert data.has_isolated_nodes()
        assert not data.has_self_loops()
        assert data.is_undirected()


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
