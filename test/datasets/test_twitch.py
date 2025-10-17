from torch_geometric.datasets import Twitch
from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_twitch():
    dataset = Twitch(root='./data/Twitch', name='ES')
    data = dataset[0]
    assert data.x.size() == (4648, 128)
    assert data.edge_index.size() == (2, 59382)
