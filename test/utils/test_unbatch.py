import torch

from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch, unbatch_edge_index


def test_unbatch():
    src = torch.arange(10)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])

    out = unbatch(src, batch)
    assert len(out) == 5
    for i in range(len(out)):
        assert torch.equal(out[i], src[batch == i])


def test_unbatch_edge_index():
    a = Data(
        edge_index=torch.randint(0, 100, [2, 100], dtype=torch.int64),
        num_nodes=100,
    )
    b = Data(
        edge_index=torch.randint(10, 50, [2, 70], dtype=torch.int64),
        num_nodes=50,
    )
    c = Batch.from_data_list([a, b])

    a_edge_index, b_edge_index = unbatch_edge_index(c.edge_index, c.batch)

    assert torch.equal(a.edge_index, a_edge_index)
    assert torch.equal(b.edge_index, b_edge_index)
