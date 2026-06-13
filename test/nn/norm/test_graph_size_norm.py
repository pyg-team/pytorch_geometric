import torch

from torch_geometric.nn import GraphSizeNorm


def test_graph_size_norm():
    x = torch.randn(100, 16)
    batch = torch.repeat_interleave(torch.full((10, ), 10, dtype=torch.long))

    norm = GraphSizeNorm()
    assert str(norm) == 'GraphSizeNorm()'

    out = norm(x, batch)
    assert out.size() == (100, 16)
