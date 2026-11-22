import copy

import pytest
import torch

from torch_geometric.contrib.nn.models.relgt_encoder import RelGTTokenEncoder


@pytest.fixture(params=[
    pytest.param(torch.device('cpu'), id='cpu'),
    pytest.param(
        torch.device('cuda:0'), id='cuda:0', marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='CUDA not available')),
])
def device(request):
    return request.param


def inputs(device):
    generator = torch.Generator().manual_seed(42)
    data = dict(
        x=torch.randn(2, 3, 16, generator=generator),
        node_type=torch.tensor([[0, 1, 2], [1, 0, 2]]),
        hop=torch.tensor([[0, 1, 2], [0, 1, 3]]),
        age=torch.tensor([[0., 2., -1.], [0., 1., 4.]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                                 [1, 0, 2, 1, 4, 3, 5, 4]]),
        pe_noise=torch.randn(2, 3, 1, generator=generator),
    )
    return {key: value.to(device) for key, value in data.items()}


def test_relgt_encoder_gradients(device):
    torch.manual_seed(0)
    encoder = RelGTTokenEncoder(16, 2).to(device)
    data = inputs(device)
    data['x'].requires_grad_()
    out = encoder(**data)
    assert out.shape == (2, 3, 16)
    out.square().mean().backward()
    assert data['x'].grad is not None
    assert data['x'].grad.abs().sum() > 0
    for module in [
            encoder.type_embedding, encoder.hop_embedding,
            encoder.time_projection, encoder.pe_encoder.input_proj,
            encoder.in_mixture[0]
    ]:
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert encoder.time_mask.grad.abs().sum() > 0


@pytest.mark.parametrize('training', [False, True])
def test_relgt_encoder_requires_explicit_noise(device, training):
    encoder = RelGTTokenEncoder(16, 2).to(device).train(training)
    data = inputs(device)
    data['pe_noise'] = None
    state = copy.deepcopy(encoder.state_dict())
    with pytest.raises(ValueError, match='pe_noise.*required'):
        encoder(**data)
    for key, value in state.items():
        assert torch.equal(value, encoder.state_dict()[key])


def test_relgt_encoder_eval_has_no_side_effects(device):
    encoder = RelGTTokenEncoder(16, 2).to(device).eval()
    data = inputs(device)
    state = copy.deepcopy(encoder.state_dict())
    rng = torch.get_rng_state().clone()
    cuda_rng = (torch.cuda.get_rng_state(device).clone()
                if device.type == 'cuda' else None)
    first = encoder(**data)
    second = encoder(**data)
    assert torch.equal(first, second)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for key, value in state.items():
        assert torch.equal(value, encoder.state_dict()[key])


def test_relgt_encoder_permutation(device):
    encoder = RelGTTokenEncoder(16, 2).to(device).eval()
    data = inputs(device)
    out = encoder(**data)
    permutation = torch.tensor([0, 2, 1], device=device)
    flattened = torch.cat([permutation, permutation + 3])
    inverse = torch.argsort(flattened)
    permuted = {
        key: value[:, permutation] if key != 'edge_index' else inverse[value]
        for key, value in data.items()
    }
    torch.testing.assert_close(encoder(**permuted), out[:, permutation])


def test_relgt_encoder_isolated_singleton(device):
    encoder = RelGTTokenEncoder(16, 2).to(device).eval()
    data = inputs(device)
    single = {
        key: value[:1, :1] if key != 'edge_index' else value[:, :0]
        for key, value in data.items()
    }
    assert encoder(**single).shape == (1, 1, 16)
    encoder.train()
    with pytest.raises(ValueError, match='at least two token'):
        encoder(**single)


def test_relgt_encoder_context_isolation(device):
    encoder = RelGTTokenEncoder(16, 2).to(device).eval()
    data = inputs(device)
    expected = encoder(**data)[:1]
    single = {
        key: value[:1]
        for key, value in data.items() if key != 'edge_index'
    }
    edges = data['edge_index']
    single['edge_index'] = edges[:, (edges < 3).all(dim=0)]
    torch.testing.assert_close(encoder(**single), expected)
    changed = {key: value.clone() for key, value in data.items()}
    changed['x'][1].add_(10.)
    changed['age'][1].add_(100.)
    changed['pe_noise'][1].mul_(2.)
    torch.testing.assert_close(encoder(**changed)[:1], expected)


@pytest.mark.parametrize('ablation', ['time', 'structure'])
def test_relgt_encoder_ablation(device, ablation):
    encoder = RelGTTokenEncoder(16, 2, ablation=ablation).to(device).eval()
    data = inputs(device)
    if ablation == 'structure':
        data['pe_noise'] = None
        assert encoder.pe_encoder is None
    else:
        assert encoder.time_mask is None
    assert encoder.in_mixture[0].in_features == 4 * 16
    assert encoder(**data).shape == (2, 3, 16)
    changed = copy.deepcopy(data)
    if ablation == 'time':
        changed['age'] = torch.full_like(changed['age'], 100.)
    else:
        changed['edge_index'] = changed['edge_index'][:, :0]
    torch.testing.assert_close(encoder(**data), encoder(**changed))


def test_relgt_encoder_checkpoint_and_reset(device):
    encoder = RelGTTokenEncoder(16, 2).to(device)
    data = inputs(device)
    encoder(**data)
    encoder.eval()
    restored = RelGTTokenEncoder(16, 2).to(device).eval()
    restored.load_state_dict(encoder.state_dict())
    torch.testing.assert_close(encoder(**data), restored(**data))
    with torch.no_grad():
        for conv in encoder.pe_encoder.conv:
            conv.eps.fill_(1.)
    encoder.reset_parameters()
    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            assert torch.equal(module.running_mean,
                               torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var,
                               torch.ones_like(module.running_var))
            assert module.num_batches_tracked == 0
    for conv in encoder.pe_encoder.conv:
        assert conv.eps.item() == 0.
    torch.manual_seed(9)
    encoder.reset_parameters()
    state = copy.deepcopy(encoder.state_dict())
    torch.manual_seed(9)
    encoder.reset_parameters()
    for key, value in state.items():
        assert torch.equal(value, encoder.state_dict()[key])


@pytest.mark.parametrize('field,value,message', [
    ('node_type', torch.zeros(2, 3), 'dtype torch.long'),
    ('node_type', torch.full((2, 3), 3), 'supported range'),
    ('hop', torch.full((2, 3), 4), 'supported range'),
    ('age', torch.full((2, 3), -2.), 'nonnegative'),
    ('age', torch.full((2, 3), float('nan')), 'finite'),
    ('edge_index', torch.tensor([[0], [3]]), 'connect contexts'),
    ('edge_index', torch.tensor([[0], [6]]), 'invalid token'),
    ('pe_noise', torch.zeros(2, 3, 2), 'shape'),
])
def test_relgt_encoder_invalid_input(device, field, value, message):
    encoder = RelGTTokenEncoder(16, 2).to(device)
    data = inputs(device)
    data[field] = value.to(device)
    state = copy.deepcopy(encoder.state_dict())
    with pytest.raises((ValueError, TypeError), match=message):
        encoder(**data)
    for key, previous in state.items():
        assert torch.equal(previous, encoder.state_dict()[key])


@pytest.mark.parametrize('kwargs', [
    dict(channels=6, num_node_types=2),
    dict(channels=0, num_node_types=2),
    dict(channels=16, num_node_types=0),
    dict(channels=16, num_node_types=2, pe_layers=0),
    dict(channels=16, num_node_types=2, max_hop=-1),
    dict(channels=16, num_node_types=2, ablation='features'),
])
def test_relgt_encoder_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RelGTTokenEncoder(**kwargs)
