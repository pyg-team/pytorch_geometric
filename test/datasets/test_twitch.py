import pytest

from torch_geometric.datasets import Twitch
from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
@pytest.mark.parametrize('name, n_dim, e_dim', [
    ('ENGB', 7126, 35324),
    ('ES', 4648, 59382),
    ('FR', 6551, 112666),
    ('PTBR', 1912, 31299),
    ('RU', 4385, 37304),
    ('DE', 9498, 153138),
])
def test_twitch(name: str, n_dim: int, e_dim: int):
    dataset = Twitch(root='./data/Twitch', name=name)
    data = dataset[0]
    assert data.x.size() == (n_dim, 128)
    assert data.edge_index.size() == (2, e_dim)
