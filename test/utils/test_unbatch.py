import pytest
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
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5],
    ])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    edge_indices = unbatch_edge_index(edge_index, batch)
    assert edge_indices[0].tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert edge_indices[1].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]


@pytest.mark.parametrize('has_edges', [
    [True, False, True, False],
    [False, True],
    [False, False],
])
@pytest.mark.parametrize('explicit_batch_size', [False, True])
def test_unbatch_edge_index_with_edgeless_graphs(has_edges,
                                                 explicit_batch_size):
    edge_index = torch.tensor([[0, 1], [1, 0]])
    empty = torch.empty((2, 0), dtype=torch.long)
    graphs = [
        Data(edge_index=edge_index if has_edge else empty,
             num_nodes=2 if has_edge else 1) for has_edge in has_edges
    ]
    batch = Batch.from_data_list(graphs)
    expected = [graph.edge_index for graph in graphs]
    batch_size = None
    if explicit_batch_size:
        batch_size = batch.num_graphs + 2
        expected.extend([empty, empty])

    out = unbatch_edge_index(batch.edge_index, batch.batch,
                             batch_size=batch_size)
    assert len(out) == len(expected)
    for actual, edge_index in zip(out, expected):
        assert torch.equal(actual, edge_index)
