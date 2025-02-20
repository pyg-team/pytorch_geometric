from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_movielens25(get_dataset):
    dataset = get_dataset(name='Movielens25')
    assert str(dataset) == 'Movielens25'
    assert len(dataset) == 2

    data = dataset[0]
    assert len(data) == 3
    assert data['movie', 'ratedby',
                'user']['edge_index'].size() == (2, 16063558)
    assert data['users']['num_users'] == 32848
    assert data['movie']['x'].size() == (58429, 16)
