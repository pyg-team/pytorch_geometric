import pytest
import torch

from torch_geometric.contrib.nn.aggr import SSMAAggregation
from torch_geometric.nn.conv import GATConv, GCNConv


@pytest.fixture
def x() -> torch.Tensor:
    return torch.rand(12, 8)


@pytest.fixture()
def edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])


@pytest.fixture
def index(edge_index: torch.Tensor) -> torch.Tensor:
    return edge_index[1]


def test_ssma_verify_pre_hook(x: torch.Tensor, index: torch.Tensor):
    ssma_agg = SSMAAggregation(in_dim=x.size(1), num_neighbors=4,
                               mlp_compression=1.0, use_attention=True)
    with pytest.raises(AssertionError):
        ssma_agg(x=x, index=index)

    ssma_agg = SSMAAggregation(in_dim=x.size(1), num_neighbors=4,
                               mlp_compression=1.0, use_attention=False)
    ssma_agg(x=x, index=index)


def test_ssma_pre_hook_reset(x: torch.Tensor, index: torch.Tensor):
    ssma_agg = SSMAAggregation(in_dim=x.size(1), num_neighbors=4,
                               mlp_compression=1.0, use_attention=True)
    ssma_agg._pre_hook_run = True
    ssma_agg._edge_attention_ste = torch.ones(len(index), 1, 4)
    ssma_agg(x=x, index=index)

    with pytest.raises(AssertionError):
        ssma_agg(x=x, index=index)


def test_ssma_res():
    index = torch.tensor([0, 0, 1, 1])
    x = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0.0, 1.0]])
    ssma_agg = SSMAAggregation(in_dim=x.size(1), num_neighbors=4,
                               mlp_compression=1.0, use_attention=False)
    # Fix MLP
    ssma_agg._mlp.bias.data = torch.zeros_like(ssma_agg._mlp.bias)
    ssma_agg._mlp.weight.data = torch.eye(n=ssma_agg._mlp.weight.size(0),
                                          m=ssma_agg._mlp.weight.size(1))

    ssma_agg._pre_hook_run = True
    x_gen = ssma_agg(x=x, index=index)

    x_expected = torch.tensor([[0.47023, 0], [0.0, 0]])

    assert torch.allclose(x_gen, x_expected, atol=1e-5)

    x = torch.zeros_like(x)
    ssma_agg._pre_hook_run = True
    x_gen = ssma_agg(x=x, index=index)
    assert torch.allclose(x_gen, torch.zeros_like(x_gen), atol=1e-6)


@pytest.mark.parametrize(("n_neighbors", "mlp_compression", "use_attention"),
                         [(2, 0.1, True), (3, 0.2, False), (4, 1.0, True)])
@pytest.mark.parametrize("layer", [GCNConv, GATConv])
def test_ssma_integration(x, edge_index, n_neighbors, mlp_compression,
                          use_attention, layer):
    ssma_agg = SSMAAggregation(in_dim=x.size(1), num_neighbors=n_neighbors,
                               mlp_compression=mlp_compression,
                               use_attention=use_attention)

    layer_kwargs = dict(in_channels=x.size(1), out_channels=x.size(1))
    layer_kwargs.update(dict(aggr=ssma_agg))

    layer = layer(**layer_kwargs)
    layer.register_propagate_forward_pre_hook(ssma_agg.pre_aggregation_hook)

    out = layer(x, edge_index)

    assert out.shape == x.shape
