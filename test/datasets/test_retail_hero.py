from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_retail_hero(get_dataset):
    dataset = get_dataset(name='RetailHero')
    assert str(dataset) == 'RetailHero()'
    assert len(dataset) == 2

    data = dataset[0]
    assert len(data) == 3
    assert data['user', 'buys',
                'product']['edge_index'].size() == (2, 14543339)
    assert data['products']['num_products'] == 40542
    assert data['user']['x'].size() == (180653, 7)
