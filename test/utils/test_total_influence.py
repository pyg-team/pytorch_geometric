import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import total_influence


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 6)
        self.conv2 = GCNConv(6, 7)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(x1, edge_index)
        return x2


def get_data():
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    return Data(x=x, edge_index=edge_index)


def test_total_influence_smoke():
    max_hops = 2
    num_samples = 4
    data = get_data()
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


def test_total_influence_normalize_average():
    max_hops = 2
    data = get_data()
    model = GNN()

    # Sampling all nodes makes the hop-wise mean invariant to the order in
    # which seed nodes are drawn:
    I_norm, _ = total_influence(model, data, max_hops=max_hops, average=True,
                                normalize=True)
    I_raw, _ = total_influence(model, data, max_hops=max_hops, average=True,
                               normalize=False)

    # Normalization is relative to the hop-0 influence:
    assert torch.isclose(I_norm[0], torch.ones(1))
    assert torch.allclose(I_norm, I_raw / I_raw[0])
    assert not torch.allclose(I_norm, I_raw)


def test_total_influence_normalize_no_average():
    max_hops = 2
    num_samples = 4
    data = get_data()
    model = GNN()

    def run(normalize):
        # Seed so that both calls draw the same set of seed nodes:
        torch.manual_seed(12345)
        return total_influence(model, data, max_hops=max_hops,
                               num_samples=num_samples, average=False,
                               normalize=normalize)[0]

    I_norm = run(normalize=True)
    I_raw = run(normalize=False)

    assert I_norm.shape == torch.Size([num_samples, max_hops + 1])

    # `normalize` must be respected in non-averaged mode as well, normalizing
    # every row by its own hop-0 influence:
    assert torch.allclose(I_norm[:, 0], torch.ones(num_samples))
    assert torch.allclose(I_norm, I_raw / I_raw[:, 0:1])
    assert not torch.allclose(I_norm, I_raw)
