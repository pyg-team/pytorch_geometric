from torch_geometric.datasets import GitHub
from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_github():
    dataset = GitHub(root='./data/GitHub')
    data = dataset[0]
    assert data.x.size() == (37700, 128)
    assert data.edge_index.size() == (2, 289003)
