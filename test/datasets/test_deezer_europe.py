from torch_geometric.datasets import DeezerEurope
from torch_geometric.testing import onlyFullTest, onlyOnline, withPackage


@onlyOnline
@onlyFullTest
@withPackage('pandas')
def test_deezer_europe():
    dataset = DeezerEurope(root='./data/DeezerEurope')
    data = dataset[0]
    assert data.x.size() == (28281, 128)
    assert data.edge_index.size() == (2, 92752)
