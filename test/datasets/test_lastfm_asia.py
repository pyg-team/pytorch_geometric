from torch_geometric.datasets import LastFMAsia
from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_lastfm_asia():
    dataset = LastFMAsia(root='./data/LastFMAsia')
    data = dataset[0]
    assert data.x.size() == (7624, 128)
    assert data.edge_index.size() == (2, 27806)
