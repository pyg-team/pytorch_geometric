import copy

import pytest
import torch

from torch_geometric.contrib.nn.models._relgt_local import _RelGTLocal


@pytest.fixture(params=[
    pytest.param(torch.device('cpu'), id='cpu'),
    pytest.param(
        torch.device('cuda:0'), id='cuda:0', marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='CUDA not available')),
])
def device(request):
    return request.param


def test_relgt_local_training(device):
    torch.manual_seed(42)
    local = _RelGTLocal(16, num_layers=2).to(device)
    tokens = torch.randn(2, 3, 16, device=device, requires_grad=True)
    out = local(tokens)
    assert out.shape == (2, 16)
    out.square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum() > 0
    for layer in local.layers:
        for projection in [layer.q_proj, layer.k_proj, layer.v_proj]:
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert projection.weight.grad.abs().sum() > 0
        assert layer.ffn.bn_in.num_batches_tracked == 1
        assert layer.ffn.bn_out.num_batches_tracked == 1


@pytest.mark.parametrize('context_size', [1, 4])
def test_relgt_local_eval_batch_one(device, context_size):
    local = _RelGTLocal(16, num_layers=2, dropout=0.8,
                        attn_dropout=0.8).to(device).eval()
    tokens = torch.randn(1, context_size, 16, device=device)
    state = copy.deepcopy(local.state_dict())
    rng = torch.get_rng_state().clone()
    cuda_rng = (torch.cuda.get_rng_state(device).clone()
                if device.type == 'cuda' else None)
    first, second = local(tokens), local(tokens)
    assert first.shape == (1, 16)
    assert torch.equal(first, second)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for key, value in state.items():
        assert torch.equal(value, local.state_dict()[key])


def test_relgt_local_seed_plus_weighted_neighbors(device):
    local = _RelGTLocal(4, num_layers=1, heads=1).to(device).eval()
    local.att_embeddings_nope = torch.nn.Identity()
    local.layers = torch.nn.ModuleList()
    local.final_ln = torch.nn.Identity()
    with torch.no_grad():
        local.attn_layer.weight.zero_()
        local.attn_layer.weight[0, 4] = 1.
        local.attn_layer.bias.zero_()
    tokens = torch.tensor(
        [[[3., 4., 5., 6.], [0., 2., 4., 6.], [1., 3., 5., 7.]]],
        device=device)
    weights = torch.tensor([0., 1.], device=device).softmax(dim=0)
    expected = tokens[:, 0] + (tokens[:, 1:] * weights[None, :, None]).sum(1)
    torch.testing.assert_close(local(tokens), expected)
    torch.testing.assert_close(local(tokens[:, :1]), tokens[:, 0])


def test_relgt_local_permutation_and_context_isolation(device):
    local = _RelGTLocal(16, num_layers=2).to(device).eval()
    tokens = torch.randn(2, 4, 16, device=device)
    out = local(tokens)
    torch.testing.assert_close(local(tokens[:, [0, 3, 1, 2]]), out)
    torch.testing.assert_close(local(tokens[:1]), out[:1])
    changed = tokens.clone()
    changed[1].normal_()
    torch.testing.assert_close(local(changed)[:1], out[:1])


def test_relgt_local_checkpoint_and_reset(device):
    local = _RelGTLocal(16, num_layers=2).to(device)
    tokens = torch.randn(2, 3, 16, device=device)
    local(tokens)
    local.eval()
    restored = _RelGTLocal(16, num_layers=2).to(device).eval()
    restored.load_state_dict(local.state_dict())
    torch.testing.assert_close(local(tokens), restored(tokens))
    local.reset_parameters()
    for module in local.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            assert module.num_batches_tracked == 0
            assert torch.equal(module.running_mean,
                               torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var,
                               torch.ones_like(module.running_var))


def test_relgt_local_invalid_inputs_leave_state_unchanged(device):
    local = _RelGTLocal(16, num_layers=2).to(device)
    state = copy.deepcopy(local.state_dict())
    for tokens in [
            torch.empty(0, 3, 16),
            torch.empty(2, 0, 16),
            torch.zeros(2, 3, 15),
            torch.zeros(1, 1, 16),
            torch.full((2, 3, 16), float('inf'))
    ]:
        with pytest.raises(ValueError):
            local(tokens.to(device))
    for key, value in state.items():
        assert torch.equal(value, local.state_dict()[key])


@pytest.mark.parametrize('kwargs', [
    dict(channels=0),
    dict(channels=16, num_layers=0),
    dict(channels=16, heads=3),
    dict(channels=16, dropout=1.),
    dict(channels=16, attn_dropout=-0.1),
])
def test_relgt_local_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        _RelGTLocal(**kwargs)
