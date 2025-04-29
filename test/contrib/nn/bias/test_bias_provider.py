import pytest
import torch

from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


class DummyProvider(torch.nn.Module):
    """Dummy bias provider with a learnable scale parameter."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch: Batch) -> torch.Tensor:
        num_graphs = int(batch.batch.max()) + 1
        num_nodes = int(torch.bincount(batch.batch).max())
        bias = (
            torch.ones(num_graphs, 4, num_nodes, num_nodes) * 3.14 * self.scale
        )
        return bias


@pytest.fixture
def two_graph_batch():
    """Batch of two graphs with 3 nodes each and 8 features."""
    data_list = []
    for _ in range(2):
        x = torch.randn(3, 8)
        data_list.append(
            Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
        )
    return Batch.from_data_list(data_list)


def test_bias_provider_affects_transformer(two_graph_batch):
    """Outputs differ when bias provider is applied."""
    model_with = GraphTransformer(
        hidden_dim=8,
        num_encoder_layers=1,
        attn_bias_providers=[DummyProvider()]
    )
    model_without = GraphTransformer(hidden_dim=8, num_encoder_layers=1)
    out_with = model_with(two_graph_batch)["logits"]
    out_without = model_without(two_graph_batch)["logits"]

    assert out_with.shape == out_without.shape
    assert not torch.allclose(out_with, out_without, rtol=1e-5, atol=1e-5)


def test_bias_provider_backward(two_graph_batch):
    """Bias provider parameters receive gradients."""
    model = GraphTransformer(
        hidden_dim=8,
        num_encoder_layers=1,
        attn_bias_providers=[DummyProvider()]
    )
    out = model(two_graph_batch)["logits"]
    out.sum().backward()

    grad = model.attn_bias_providers[0].scale.grad
    assert grad is not None
    assert not torch.allclose(grad, torch.zeros_like(grad))
