import torch

from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


class DummyProvider(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Add a learnable parameter to test gradient flow
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch) -> torch.Tensor:
        num_graphs = int(batch.batch.max()) + 1
        num_nodes = max(torch.bincount(batch.batch))
        scaled_bias = torch.ones(num_graphs, 4, num_nodes, num_nodes) \
            * 3.14 * self.scale
        return scaled_bias


def test_bias_provider():
    # Create two tiny graphs
    data_list = []
    for _ in range(2):
        x = torch.randn(3, 8)  # 3 nodes, 8 features
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list)

    # Create two models - one with provider, one without
    model_with_bias = GraphTransformer(
        hidden_dim=8,
        num_encoder_layers=1,
        attn_bias_providers=[DummyProvider()]
    )
    model_without_bias = GraphTransformer(hidden_dim=8, num_encoder_layers=1)

    # Compare outputs
    out_with_bias = model_with_bias(batch)["logits"]
    out_without_bias = model_without_bias(batch)["logits"]

    # Use a smaller tolerance for the comparison
    assert not torch.allclose(
        out_with_bias, out_without_bias, rtol=1e-5, atol=1e-5), \
        "Outputs should differ when using bias provider"

    # Test backward pass
    loss = out_with_bias.sum()
    loss.backward()

    # Check gradient flow
    bias_provider = model_with_bias.attn_bias_providers[0]
    assert bias_provider.scale.grad is not None, \
        "Bias provider parameters should receive gradients"
