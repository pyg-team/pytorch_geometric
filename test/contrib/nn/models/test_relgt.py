import copy
import math

import pytest
import torch

from torch_geometric.contrib.nn.models.relgt import (
    RelGT,
    _RelGTGlobal,
    _RelGTMemory,
)
from torch_geometric.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('mode', ['full', 'local'])
@pytest.mark.parametrize('out_channels', [1, 4])
def test_relgt_training_and_checkpoint(device, mode, out_channels):
    torch.manual_seed(7)
    model = RelGT(8, out_channels, 10, num_layers=1, heads=2, num_centroids=4,
                  global_channels=4, mode=mode, attn_dropout=0.2).to(device)
    tokens = torch.randn(3, 5, 8, device=device, requires_grad=True)
    ids = torch.tensor([1, 2, 1], device=device)
    pred = model(tokens, ids)
    assert pred.shape == (3, out_channels)
    pred.square().mean().backward()
    assert tokens.grad.isfinite().all()
    assert tokens.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters())
    assert all(p.grad.isfinite().all() for p in model.parameters())
    model.eval()
    state = copy.deepcopy(model.state_dict())
    single = model(tokens[:1], ids[:1])
    assert single.shape == (1, out_channels)
    assert torch.equal(single, model(tokens[:1], ids[:1]))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    restored = RelGT(8, out_channels, 10, num_layers=1, heads=2,
                     num_centroids=4, global_channels=4, mode=mode,
                     attn_dropout=0.2).to(device).eval()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(tokens[:1], ids[:1]), single)
    if mode == 'full':
        memory = model.global_module.memory
        assert memory.ema_count.sum() > 0
        assert memory.population.sum() == 10
        assert torch.equal(memory.population,
                           memory.assignment.bincount(minlength=4))


@withCUDA
def test_population_weighted_attention_reads_old_memory(device):
    torch.manual_seed(8)
    layer = _RelGTGlobal(4, 2, 5, 3, 2, 0.99, 0.).to(device)
    with torch.no_grad():
        layer.memory.assignment.copy_(
            torch.tensor([0, 0, 2, 2, 2], device=device))
        layer.memory.population.copy_(torch.tensor([2, 0, 3], device=device))
    seed = torch.randn(2, 4, device=device)
    x = layer.project(seed)
    centroids = layer.memory.output.clone()
    q, k, v = layer.query(x), layer.key(centroids), layer.value(centroids)
    expected = []
    for sample in range(2):
        values = []
        for head in range(2):
            sl = slice(head * 2, (head + 1) * 2)
            logits = (q[sample, sl] @ k[:, sl].T) / math.sqrt(4)
            logits = logits + torch.tensor([2., 0., 3.], device=device).log()
            values.append(logits.softmax(0) @ v[:, sl])
        expected.append(torch.cat(values))
    actual = layer(seed, torch.tensor([0, 1], device=device))
    torch.testing.assert_close(actual, torch.stack(expected))
    assert not torch.equal(centroids, layer.memory.output)
    actual.sum().backward()  # EMA updates must not invalidate autograd.


@withCUDA
def test_ema_formula_and_duplicate_assignments(device):
    memory = _RelGTMemory(4, 2, 2, decay=0.5).to(device)
    with torch.no_grad():
        memory.centroids.copy_(
            torch.tensor([[-1., -1.], [1., 1.]], device=device))
        memory.ema_sum.zero_()
        memory.assignment.zero_()
        memory.population.copy_(torch.tensor([4, 0], device=device))
    x = torch.tensor([[-2., -2.], [2., 2.]], device=device)
    normalized = x / math.sqrt(4. + 1e-5)
    expected_sum = normalized * 0.5
    memory.update(x, torch.tensor([1, 1], device=device))
    torch.testing.assert_close(memory.ema_count,
                               torch.tensor([0.5, 0.5], device=device))
    torch.testing.assert_close(memory.ema_sum, expected_sum)
    torch.testing.assert_close(memory.centroids, normalized)
    # Last occurrence wins while both observations contributed to the EMA.
    assert memory.assignment.tolist() == [0, 1, 0, 0]
    assert memory.population.tolist() == [3, 1]
    torch.testing.assert_close(memory.norm.running_mean,
                               torch.zeros(2, device=device))
    torch.testing.assert_close(memory.norm.running_var,
                               torch.full((2, ), 1.7, device=device))
    torch.testing.assert_close(memory.output,
                               normalized * math.sqrt(1.7 + 1e-5))


@withCUDA
def test_relgt_rejects_invalid_input_before_state_changes(device):
    model = RelGT(8, 1, 5, num_layers=1, heads=2, num_centroids=4,
                  global_channels=4).to(device)
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='B >= 2'):
        model(torch.randn(1, 3, 8, device=device),
              torch.tensor([0], device=device))
    with pytest.raises(ValueError, match='outside'):
        model(torch.randn(2, 3, 8, device=device),
              torch.tensor([0, 5], device=device))
    with pytest.raises(ValueError, match='seed_id'):
        model(torch.randn(2, 3, 8, device=device))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


@withCUDA
def test_invalid_population_and_memory_reset(device):
    model = RelGT(8, 1, 5, num_layers=1, heads=2, num_centroids=4,
                  global_channels=4).to(device).eval()
    memory = model.global_module.memory
    memory.population.zero_()
    with pytest.raises(ValueError, match='nonzero'):
        model(torch.randn(1, 3, 8, device=device))
    parameters = {k: p.clone() for k, p in model.named_parameters()}
    model.reset_memory()
    assert memory.population.sum() == 5
    assert all(
        torch.equal(parameters[k], p) for k, p in model.named_parameters())
    assert torch.equal(memory.population,
                       memory.assignment.bincount(minlength=4))
    assert model(torch.randn(1, 3, 8, device=device)).isfinite().all()
