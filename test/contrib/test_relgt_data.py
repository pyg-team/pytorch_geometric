from dataclasses import replace
from pathlib import Path

import pytest
import torch

from examples.contrib.relgt.data import (
    PredictionSample,
    RelGTGraph,
    collate_contexts,
    sample_context,
)


def graph() -> RelGTGraph:
    # Rows 0 and 3 are audited static rows. Row 2 is a future transaction.
    return RelGTGraph(
        node_type=torch.tensor([0, 1, 1, 2]),
        node_id=torch.tensor([0, 0, 1, 0]),
        time=torch.tensor([0, 5, 15, 0]),
        time_known=torch.tensor([False, True, True, False]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
    )


def sample(time: int = 10, row: str = 'row-0') -> PredictionSample:
    return PredictionSample('dataset-sha', 'task-sha', 'train', row, 0, time)


def test_repeated_entity_at_different_times(tmp_path: Path) -> None:
    earlier = sample_context(graph(), sample(), context_size=3,
                             cache_dir=tmp_path)
    later = sample_context(graph(), sample(20, 'row-1'), context_size=3,
                           cache_dir=tmp_path)
    assert earlier.sample_id != later.sample_id
    assert earlier.node_index.tolist() == [0, 1, 1]
    assert set(later.node_index.tolist()) == {0, 1, 2}
    assert len(list(tmp_path.glob('*.pt'))) == 2
    assert earlier.age[1] == pytest.approx(5 / 86400)
    assert later.age[later.node_index == 1] == pytest.approx(15 / 86400)


def test_duplicate_occurrence_edges() -> None:
    context = sample_context(graph(), sample(), context_size=4)
    assert context.node_index.tolist() == [0, 1, 1, 1]
    assert set(map(tuple,
                   context.edge_index.t().tolist())) == {(0, 1), (0, 2),
                                                         (0, 3), (1, 0),
                                                         (2, 0), (3, 0)}


def test_eligible_fallback_and_seed_repeat() -> None:
    data = replace(graph(), edge_index=torch.empty((2, 0), dtype=torch.long))
    context = sample_context(data, sample(0), context_size=4)
    assert context.node_index.tolist() == [0, 3, 3, 3]
    assert context.hop.tolist() == [0, 3, 3, 3]
    assert context.edge_index.shape == (2, 0)
    data = replace(data, time_known=torch.tensor([False, True, True, True]),
                   time=torch.tensor([0, 5, 15, 1]))
    context = sample_context(data, sample(0), context_size=3)
    assert context.node_index.tolist() == [0, 0, 0]
    assert context.hop.tolist() == [0, 3, 3]
    assert context.age.tolist() == [0, 0, 0]


def test_future_changes_do_not_change_context() -> None:
    data = graph()
    before = sample_context(data, sample(), context_size=5, max_one_hop=1)
    changed = replace(data, time=torch.tensor([0, 5, 999, 0]),
                      edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]))
    after = sample_context(changed, sample(), context_size=5, max_one_hop=1)
    for name in ('node_index', 'node_type', 'hop', 'age', 'edge_index'):
        assert torch.equal(getattr(before, name), getattr(after, name))


def test_cutoff_and_two_hop() -> None:
    data = replace(graph(), edge_index=torch.tensor([[0, 1], [1, 2]]))
    context = sample_context(data, sample(15), context_size=3)
    hops = dict(zip(context.node_index.tolist(), context.hop.tolist()))
    assert hops == {0: 0, 1: 1, 2: 2}
    assert context.age[context.node_index == 2].item() == 0
    # A future first hop cannot expose an otherwise static two-hop row.
    data = replace(data, edge_index=torch.tensor([[0, 2], [2, 3]]))
    context = sample_context(data, sample(10), context_size=3)
    assert context.hop[1:].tolist() == [3, 3]
    assert 2 not in context.node_index


def test_context_size_one_and_real_self_relation() -> None:
    data = replace(graph(), edge_index=torch.tensor([[0], [0]]))
    context = sample_context(data, sample(), context_size=1)
    assert context.node_index.tolist() == [0]
    assert context.edge_index.tolist() == [[0], [0]]


def test_batch_offsets_global_ids_and_gradients() -> None:
    contexts = [
        sample_context(graph(), sample(), context_size=3),
        sample_context(graph(), replace(sample(), seed_index=3),
                       context_size=3)
    ]
    embeddings = torch.randn(4, 8, requires_grad=True)
    batch = collate_contexts(contexts, embeddings)
    assert batch['x'].shape == (2, 3, 8)
    assert batch['seed_id'].tolist() == [0, 3]  # Both type-local IDs are 0.
    assert batch['node_type'].shape == (2, 3)
    edge = batch['edge_index']
    assert torch.equal(edge[0] // 3, edge[1] // 3)
    batch['x'].sum().backward()
    assert embeddings.grad is not None
    assert embeddings.grad[0].abs().sum() > 0
    one = collate_contexts(contexts[:1], embeddings)
    assert one['x'].shape == (1, 3, 8)
    assert one['seed_id'].shape == (1, )


def test_cache_roundtrip_and_corruption(tmp_path: Path) -> None:
    first = sample_context(graph(), sample(), context_size=3,
                           cache_dir=tmp_path)
    again = sample_context(graph(), sample(), context_size=3,
                           cache_dir=tmp_path)
    assert torch.equal(first.node_index, again.node_index)
    assert torch.equal(first.edge_index, again.edge_index)
    path = tmp_path / (first.sample_id + '.pt')
    payload = torch.load(path, weights_only=True)
    payload['context']['age'][1] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Invalid context cache'):
        sample_context(graph(), sample(), context_size=3, cache_dir=tmp_path)


@pytest.mark.parametrize('change', ['size', 'seed', 'graph'])
def test_cache_rejects_changed_dependencies(tmp_path: Path,
                                            change: str) -> None:
    sample_context(graph(), sample(), context_size=3, cache_dir=tmp_path)
    data, size, seed = graph(), 3, 0
    if change == 'size':
        size = 4
    elif change == 'seed':
        seed = 1
    else:
        data = replace(data, time=torch.tensor([0, 6, 15, 0]))
    with pytest.raises(ValueError, match='Invalid context cache'):
        sample_context(data, sample(), context_size=size, seed=seed,
                       cache_dir=tmp_path)


def test_task_identity_not_entity_only() -> None:
    first = sample_context(graph(), sample(), context_size=3)
    for request in (replace(sample(), task_row_id='row-1'),
                    replace(sample(), split='val'),
                    replace(sample(), task_revision='new-task'),
                    replace(sample(), dataset_revision='new-dataset')):
        other = sample_context(graph(), request, context_size=3)
        assert first.sample_id != other.sample_id


def test_request_order_does_not_change_sampling() -> None:
    requests = [sample(10, 'a'), sample(20, 'b'), sample(20, 'c')]
    forwards = {
        request.task_row_id: sample_context(graph(), request, context_size=7)
        for request in requests
    }
    backwards = {
        request.task_row_id: sample_context(graph(), request, context_size=7)
        for request in reversed(requests)
    }
    for key in forwards:
        assert torch.equal(forwards[key].node_index, backwards[key].node_index)
        assert torch.equal(forwards[key].edge_index, backwards[key].edge_index)


def test_truncated_cache_fails(tmp_path: Path) -> None:
    context = sample_context(graph(), sample(), context_size=3,
                             cache_dir=tmp_path)
    (tmp_path / (context.sample_id + '.pt')).write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Invalid context cache'):
        sample_context(graph(), sample(), context_size=3, cache_dir=tmp_path)


@pytest.mark.parametrize('change', ['duplicate', 'edge', 'time', 'seed'])
def test_invalid_inputs(change: str) -> None:
    data, request = graph(), sample()
    if change == 'duplicate':
        data = replace(data, node_id=torch.tensor([0, 0, 0, 0]))
    elif change == 'edge':
        data = replace(data, edge_index=torch.tensor([[0], [4]]))
    elif change == 'time':
        data = replace(data, time=data.time.float())
    else:
        request = replace(request, seed_index=2)
    with pytest.raises(ValueError):
        sample_context(data, request, context_size=3)
