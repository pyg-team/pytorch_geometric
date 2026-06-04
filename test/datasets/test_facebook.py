from torch_geometric.datasets import FacebookPagePage
from torch_geometric.testing import onlyFullTest, onlyOnline, withPackage


@onlyOnline
@onlyFullTest
@withPackage('pandas')
def test_facebook():
    dataset = FacebookPagePage(root='./data/FacebookPagePage')
    data = dataset[0]
    assert data.x.size() == (22470, 128)
    assert data.edge_index.size() == (2, 171002)
