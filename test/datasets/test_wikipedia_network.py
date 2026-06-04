from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.testing import onlyFullTest, onlyOnline, withPackage


@onlyOnline
@onlyFullTest
@withPackage('pandas')
def test_wikipedia_network():
    dataset = WikipediaNetwork(root='./data/WikipediaNetwork',
                               name='crocodile')
    data = dataset[0]
    assert data.x.size() == (11631, 128)
    assert data.edge_index.size() == (2, 180020)
