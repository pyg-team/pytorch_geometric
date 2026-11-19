import copy

import pytest
import torch

from examples.contrib.relgt.synthetic import (
    DAY,
    HORIZON,
    ORIGINS,
    BankingHistory,
    fixed_noise,
    generate_history,
    history_features,
    inactivity_target,
    metrics,
    prepare_split,
)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_simulated_history_reproducible(seed):
    before = torch.random.get_rng_state()
    history = generate_history(seed)
    repeated = generate_history(seed)
    assert torch.equal(before, torch.random.get_rng_state())
    for field in vars(history):
        first, second = getattr(history, field), getattr(repeated, field)
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second)
        else:
            assert first == second
    for origin in ORIGINS.values():
        target = inactivity_target(history, origin)
        assert 0 < target.sum() < history.num_customers
    assert ORIGINS['train'] + HORIZON <= ORIGINS['val']
    assert ORIGINS['val'] + HORIZON <= ORIGINS['test']


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_simulated_context_excludes_prediction_day_and_future(split):
    history = generate_history(4, num_customers=8)
    batch = prepare_split(history, split, seed=9, context_size=12)
    graph = history.graph()
    assert (graph.time[batch['node_index']] < ORIGINS[split] * DAY).all()
    assert batch['x'].shape == (8, 12, 8)
    assert torch.equal(batch['seed_id'], torch.arange(8))
    assert torch.equal(batch['target'],
                       inactivity_target(history, ORIGINS[split]))


def test_future_changes_cannot_change_history_features_or_context():
    history = generate_history(3, num_customers=8)
    origin = ORIGINS['train']
    altered = copy.deepcopy(history)
    future = altered.transaction_day >= origin
    # Move future events out of the target horizon and change all their values.
    altered.transaction_day[future] += HORIZON
    altered.transaction_amount[future] *= 100
    altered.transaction_account[future] = 0
    assert torch.equal(history_features(history, origin),
                       history_features(altered, origin))
    first = prepare_split(history, 'train', seed=2, context_size=12)
    second = prepare_split(altered, 'train', seed=2, context_size=12)
    for field in ['x', 'node_index', 'node_type', 'hop', 'age', 'edge_index']:
        assert torch.equal(first[field], second[field])
    assert not torch.equal(first['target'], second['target'])


def test_inactivity_uses_all_accounts_and_exact_future_window():
    history = BankingHistory(3, torch.tensor([0, 0, 1, 2]),
                             torch.tensor([1, 2, 3]),
                             torch.tensor([90, 89, 120]), torch.ones(3))
    # Day 90 counts; day 89 is history; day 120 is outside [90, 120).
    assert inactivity_target(history, 90).tolist() == [0., 1., 1.]


def test_noise_is_sample_and_time_specific_without_global_rng():
    before = torch.random.get_rng_state()
    first = fixed_noise('train', 0, 8, 12)
    assert torch.equal(first, fixed_noise('train', 0, 8, 12))
    assert not torch.equal(first, fixed_noise('val', 0, 8, 12))
    assert not torch.equal(first[0], first[1])
    assert torch.equal(before, torch.random.get_rng_state())


def test_synthetic_auc_and_class_support():
    result = metrics(torch.tensor([1., 0., 0., -1.]),
                     torch.tensor([1., 1., 0., 0.]))
    assert result['roc_auc'] == 0.875
    assert result['samples'] == 4
    assert result['inactive'] == 2
    assert metrics(torch.zeros(3), torch.ones(3))['roc_auc'] is None
