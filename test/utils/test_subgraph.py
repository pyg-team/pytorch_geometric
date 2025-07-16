import torch

from torch_geometric.nn import GCNConv, Linear
from torch_geometric.testing import withDevice, withPackage
from torch_geometric.utils import (
    bipartite_subgraph,
    get_num_hops,
    index_to_mask,
    k_hop_subgraph,
    subgraph,
)


def test_get_num_hops():
    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(3, 16, normalize=False)
            self.conv2 = GCNConv(16, 16, normalize=False)
            self.lin = Linear(16, 2)

        def forward(self, x, edge_index):
            x = torch.F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return self.lin(x)

    assert get_num_hops(GNN()) == 2


def test_subgraph():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
    ])
    edge_attr = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    idx = torch.tensor([3, 4, 5])
    mask = index_to_mask(idx, 7)
    indices = idx.tolist()

    for subset in [idx, mask, indices]:
        out = subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        assert out[0].tolist() == [[3, 4, 4, 5], [4, 3, 5, 4]]
        assert out[1].tolist() == [7.0, 8.0, 9.0, 10.0]
        assert out[2].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]

        out = subgraph(subset, edge_index, edge_attr, relabel_nodes=True)
        assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        assert out[1].tolist() == [7, 8, 9, 10]


@withDevice
@withPackage('pandas')
def test_subgraph_large_index(device):
    subset = torch.tensor([50_000_000], device=device)
    edge_index = torch.tensor([[50_000_000], [50_000_000]], device=device)
    edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
    assert edge_index.tolist() == [[0], [0]]


def test_bipartite_subgraph():
    edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
                               [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
    edge_attr = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    idx = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
    mask = (index_to_mask(idx[0], 7), index_to_mask(idx[1], 4))
    indices = (idx[0].tolist(), idx[1].tolist())
    mixed = (mask[0], idx[1])

    for subset in [idx, mask, indices, mixed]:
        out = bipartite_subgraph(subset, edge_index, edge_attr,
                                 return_edge_mask=True)
        assert out[0].tolist() == [[2, 3, 5, 5], [3, 2, 2, 3]]
        assert out[1].tolist() == [3.0, 4.0, 9.0, 10.0]
        assert out[2].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]

        out = bipartite_subgraph(subset, edge_index, edge_attr,
                                 relabel_nodes=True)
        assert out[0].tolist() == [[0, 1, 2, 2], [1, 0, 0, 1]]
        assert out[1].tolist() == [3.0, 4.0, 9.0, 10.0]


@withDevice
@withPackage('pandas')
def test_bipartite_subgraph_large_index(device):
    subset = torch.tensor([50_000_000], device=device)
    edge_index = torch.tensor([[50_000_000], [50_000_000]], device=device)

    edge_index, _ = bipartite_subgraph(
        (subset, subset),
        edge_index,
        relabel_nodes=True,
    )
    assert edge_index.tolist() == [[0], [0]]


def test_k_hop_subgraph():
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [2, 2, 4, 4, 6, 6],
    ])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=6,
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=True,
    )
    assert subset.tolist() == [2, 3, 4, 5, 6]
    assert edge_index.tolist() == [[0, 1, 2, 3], [2, 2, 4, 4]]
    assert mapping.tolist() == [4]
    assert edge_mask.tolist() == [False, False, True, True, True, True]

    edge_index = torch.tensor([
        [1, 2, 4, 5],
        [0, 1, 5, 6],
    ])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=[0, 6],
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=True,
    )
    assert subset.tolist() == [0, 1, 2, 4, 5, 6]
    assert edge_index.tolist() == [[1, 2, 3, 4], [0, 1, 4, 5]]
    assert mapping.tolist() == [0, 5]
    assert edge_mask.tolist() == [True, True, True, True]

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 4, 5],
        [2, 2, 4, 4, 2, 6, 6],
    ])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=6,
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=False,
        directed=True,
    )
    assert subset.tolist() == [2, 3, 4, 5, 6]
    assert edge_index.tolist() == [[2, 3, 4, 5], [4, 4, 6, 6]]
    assert mapping.tolist() == [4]
    assert edge_mask.tolist() == [False, False, True, True, False, True, True]
