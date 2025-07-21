import torch

from torch_geometric.contrib.nn.models import GraphTransformer


def test_bias_provider_affects_transformer(two_graph_batch,
                                           basic_encoder_config):
    """Outputs differ when bias provider is applied."""
    model_with = GraphTransformer(hidden_dim=8, num_class=2,
                                  encoder_cfg=basic_encoder_config)
    model_without = GraphTransformer(hidden_dim=8, num_class=2)
    out_with = model_with(two_graph_batch)
    out_without = model_without(two_graph_batch)

    assert out_with.shape == out_without.shape
    assert not torch.allclose(out_with, out_without, rtol=1e-5, atol=1e-5)


def test_bias_provider_backward(two_graph_batch, basic_encoder_config):
    """Bias provider parameters receive gradients."""
    model = GraphTransformer(hidden_dim=8, num_class=2,
                             encoder_cfg=basic_encoder_config)
    out = model(two_graph_batch)
    out.sum().backward()

    grad = model.attn_bias_providers[0].scale.grad
    assert grad is not None
    # Ensure the gradient is not zero
    # This checks that the bias provider is contributing to the gradient
    assert not torch.allclose(grad, torch.zeros_like(grad), rtol=1e-8,
                              atol=1e-8)
