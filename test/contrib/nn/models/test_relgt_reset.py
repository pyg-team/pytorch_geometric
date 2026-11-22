import copy

import pytest
import torch

from torch_geometric.contrib.nn.models import RelGT
from torch_geometric.testing import withCUDA


@withCUDA
@pytest.mark.parametrize('mode', ['full', 'local'])
def test_relgt_complete_reset_and_seed_repeatability(device, mode):
    torch.manual_seed(41)
    model = RelGT(16, 1, 12, num_layers=2, heads=4, num_centroids=5,
                  global_channels=8, mode=mode).to(device)
    tokens = torch.randn(4, 3, 16, device=device)
    ids = torch.tensor([1, 2, 1, 4], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    loss = model(tokens, ids).square().mean()
    loss.backward()
    optimizer.step()
    norms = [
        module for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    assert norms
    assert all(int(module.num_batches_tracked) > 0 for module in norms)

    # Force every learned tensor and running statistic away from its reset
    # value. This also catches omitted fusion/head resets after real training.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7.)
        for norm in norms:
            norm.running_mean.fill_(11.)
            norm.running_var.fill_(13.)
            norm.num_batches_tracked.fill_(17)
        if mode == 'full':
            memory = model.global_module.memory
            memory.centroids.fill_(19.)
            memory.output.fill_(23.)
            memory.ema_count.fill_(29.)
            memory.ema_sum.fill_(31.)
            memory.assignment.zero_()
            memory.population.zero_()
            memory.population[0] = 12

    torch.manual_seed(53)
    model.reset_parameters()
    first = copy.deepcopy(model.state_dict())
    assert all(not torch.all(parameter == 7.)
               for parameter in model.parameters())
    for norm in norms:
        assert torch.equal(norm.running_mean,
                           torch.zeros_like(norm.running_mean))
        assert torch.equal(norm.running_var, torch.ones_like(norm.running_var))
        assert int(norm.num_batches_tracked) == 0
    if mode == 'full':
        memory = model.global_module.memory
        assert torch.equal(memory.ema_count,
                           torch.zeros_like(memory.ema_count))
        assert not torch.all(memory.centroids == 19.)
        assert not torch.all(memory.output == 23.)
        assert not torch.all(memory.ema_sum == 31.)
        assert bool(((memory.assignment >= 0) & (memory.assignment < 5)).all())
        assert torch.equal(memory.population,
                           memory.assignment.bincount(minlength=5))
        assert int(memory.population.sum()) == 12
    else:
        assert model.global_module is None
        assert not any('memory' in name for name in first)

    # Reset consumes the supplied PyTorch RNG normally; replaying that seed
    # must restore every learned tensor and persistent global-memory buffer.
    torch.manual_seed(53)
    model.reset_parameters()
    assert all(
        torch.equal(value,
                    model.state_dict()[name]) for name, value in first.items())

    model.eval()
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        prediction = model(tokens[:1])
    assert prediction.shape == (1, 1)
    assert bool(prediction.isfinite().all())
    assert all(
        torch.equal(value,
                    model.state_dict()[name])
        for name, value in before.items())

    # The freshly reset model must also support a subsequent optimization
    # forward/backward, including a valid first EMA update in full mode.
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(tokens, ids)
    prediction.square().mean().backward()
    assert bool(prediction.isfinite().all())
    assert all(
        parameter.grad is not None and bool(parameter.grad.isfinite().all())
        for parameter in model.parameters())
    if mode == 'full':
        assert torch.equal(memory.population,
                           memory.assignment.bincount(minlength=5))
        assert int(memory.norm.num_batches_tracked) == 1
