from torch_geometric.datasets import GemsecDeezer
from torch_geometric.testing import onlyFullTest, onlyOnline, withPackage


@onlyOnline
@onlyFullTest
@withPackage('pandas')
def test_gemsec():
    dataset = GemsecDeezer(root='./data/GemsecDeezer', name='HU')
    data = dataset[0]
    assert data.edge_index.size() == (2, 222887)
