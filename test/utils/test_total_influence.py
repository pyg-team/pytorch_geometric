import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import total_influence
from torch_geometric.utils.influence import k_hop_subsets_rough


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 6)
        self.conv2 = GCNConv(6, 7)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(x1, edge_index)
        return x2


def test_total_influence_smoke():
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    max_hops = 2
    num_samples = 4
    data = Data(
        x=x,
        edge_index=edge_index,
    )
    model = GNN()
    I, R = total_influence(
        model,
        data,
        max_hops=max_hops,
        num_samples=num_samples,
    )

    assert I.shape == (max_hops + 1, )
    assert 0.0 <= R <= max_hops

    I, R = total_influence(
        model,
        data,
        max_hops=max_hops,
        num_samples=num_samples,
        average=False,
    )
    assert I.shape == torch.Size([num_samples, max_hops + 1])


def test_k_hop_subsets_rough():
    # Create a simple undirected linear graph: 0 - 1 - 2 - 3 - 4
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]])
    num_nodes = 5

    # Test 2-hop subsets from node 1
    subsets = k_hop_subsets_rough(node_idx=1, num_hops=2,
                                  edge_index=edge_index, num_nodes=num_nodes)

    # Should return [H0, H1, H2] where:
    # H0 = [1] (seed node)
    # H1 = [0, 2] (1-hop neighbors)
    # H2 = [1, 1, 3] (2-hop neighbors, may include overlaps)
    assert len(subsets) == 3
    assert subsets[0].tolist() == [1]
    assert set(subsets[1].tolist()) == {0, 2}
    assert set(subsets[2].tolist()) == {1, 3}
