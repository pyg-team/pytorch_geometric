import torch

from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch_edge_index


def test_unbatch_edge_index():
    a = Data(edge_index=torch.randint(0, 100, [2, 100], dtype=torch.int64),
             num_nodes=100)
    b = Data(edge_index=torch.randint(10, 50, [2, 70], dtype=torch.int64),
             num_nodes=50)
    c = Batch.from_data_list([a, b])

    a_edge_index, b_edge_index = unbatch_edge_index(c.edge_index, c.batch)

    assert torch.equal(a.edge_index, a_edge_index)
    assert torch.equal(b.edge_index, b_edge_index)
