from typing import List

import pytest
import torch

from torch_geometric.metrics import (
    LinkPredF1,
    LinkPredMAP,
    LinkPredMetricCollection,
    LinkPredMRR,
    LinkPredNDCG,
    LinkPredPrecision,
    LinkPredRecall,
)


@pytest.mark.parametrize('num_src_nodes', [100])
@pytest.mark.parametrize('num_dst_nodes', [1000])
@pytest.mark.parametrize('num_edges', [3000])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('k', [1, 10, 100])
def test_precision(num_src_nodes, num_dst_nodes, num_edges, batch_size, k):
    row = torch.randint(0, num_src_nodes, (num_edges, ))
    col = torch.randint(0, num_dst_nodes, (num_edges, ))
    edge_label_index = torch.stack([row, col], dim=0)

    pred = torch.rand(num_src_nodes, num_dst_nodes)
    pred[row, col] += 0.3  # Offset positive links by a little.
    pred_index_mat = pred.topk(k, dim=1)[1]

    metric = LinkPredPrecision(k)
    assert str(metric) == f'LinkPredPrecision(k={k})'

    for node_id in torch.split(torch.randperm(num_src_nodes), batch_size):
        mask = torch.isin(edge_label_index[0], node_id)

        y_batch, y_index = edge_label_index[:, mask]
        # Remap `y_batch` back to `[0, batch_size - 1]` range:
        arange = torch.empty(num_src_nodes, dtype=node_id.dtype)
        arange[node_id] = torch.arange(node_id.numel())
        y_batch = arange[y_batch]

        metric.update(pred_index_mat[node_id], (y_batch, y_index))

    out = metric.compute()
    metric.reset()

    values: List[float] = []
    for i in range(num_src_nodes):  # Naive computation per node:
        y_index = col[row == i]
        if y_index.numel() > 0:
            mask = torch.isin(pred_index_mat[i], y_index)
            precision = float(mask.sum() / k)
            values.append(precision)
    expected = torch.tensor(values).mean()
    assert torch.allclose(out, expected)

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :k - 1], edge_label_index)
    metric.compute()
    metric.reset()


def test_recall():
    pred_index_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 2, 1]])
    edge_label_weight = torch.tensor([4.0, 1.0, 2.0, 3.0, 0.5])

    metric = LinkPredRecall(k=2)
    assert str(metric) == 'LinkPredRecall(k=2)'
    metric.update(pred_index_mat, edge_label_index)
    result = metric.compute()
    assert float(result) == pytest.approx(0.5 * (2 / 3 + 0.5))

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index)
    metric.compute()
    metric.reset()

    metric = LinkPredRecall(k=2, weighted=True)
    assert str(metric) == 'LinkPredRecall(k=2, weighted=True)'
    with pytest.raises(ValueError, match="'edge_label_weight'"):
        metric.update(pred_index_mat, edge_label_index)

    metric.update(pred_index_mat, edge_label_index, edge_label_weight)
    result = metric.compute()
    assert float(result) == pytest.approx(0.5 * (5.0 / 7.0 + 3.0 / 3.5))

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index, edge_label_weight)
    metric.compute()
    metric.reset()


def test_f1():
    pred_index_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 2, 1]])

    metric = LinkPredF1(k=2)
    assert str(metric) == 'LinkPredF1(k=2)'
    metric.update(pred_index_mat, edge_label_index)
    result = metric.compute()
    assert float(result) == pytest.approx(0.6500)

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index)
    metric.compute()
    metric.reset()


def test_map():
    pred_index_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 2, 1]])

    metric = LinkPredMAP(k=2)
    assert str(metric) == 'LinkPredMAP(k=2)'
    metric.update(pred_index_mat, edge_label_index)
    result = metric.compute()
    assert float(result) == pytest.approx(0.6250)

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index)
    metric.compute()
    metric.reset()


def test_ndcg():
    pred_index_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 2, 1]])
    edge_label_weight = torch.tensor([1.0, 2.0, 0.1, 3.0, 0.5])

    metric = LinkPredNDCG(k=2)
    assert str(metric) == 'LinkPredNDCG(k=2)'
    metric.update(pred_index_mat, edge_label_index)
    result = metric.compute()
    assert float(result) == pytest.approx(0.6934264)

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index)
    metric.compute()
    metric.reset()

    metric = LinkPredNDCG(k=2, weighted=True)
    assert str(metric) == 'LinkPredNDCG(k=2, weighted=True)'
    with pytest.raises(ValueError, match="'edge_label_weight'"):
        metric.update(pred_index_mat, edge_label_index)

    metric.update(pred_index_mat, edge_label_index, edge_label_weight)
    result = metric.compute()
    metric.reset()
    assert float(result) == pytest.approx(0.7854486)

    perm = torch.randperm(edge_label_weight.size(0))
    metric.update(pred_index_mat, edge_label_index[:, perm],
                  edge_label_weight[perm])
    assert metric.compute() == result

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index, edge_label_weight)
    metric.compute()
    metric.reset()


def test_mrr():
    pred_index_mat = torch.tensor([[1, 0], [1, 2], [0, 2], [0, 1]])
    edge_label_index = torch.tensor([[0, 0, 2, 2, 3], [0, 1, 2, 1, 2]])

    metric = LinkPredMRR(k=2)
    assert str(metric) == 'LinkPredMRR(k=2)'
    metric.update(pred_index_mat, edge_label_index)
    result = metric.compute()

    assert float(result) == pytest.approx((1 + 0.5 + 0) / 3)

    # Test with `k > pred_index_mat.size(1)`:
    metric.update(pred_index_mat[:, :1], edge_label_index)
    metric.compute()
    metric.reset()


@pytest.mark.parametrize('num_src_nodes', [10])
@pytest.mark.parametrize('num_dst_nodes', [50])
@pytest.mark.parametrize('num_edges', [200])
def test_link_pred_metric_collection(num_src_nodes, num_dst_nodes, num_edges):
    metrics = [
        LinkPredMAP(k=10),
        LinkPredPrecision(k=100),
        LinkPredRecall(k=50),
        LinkPredF1(k=20),
        LinkPredMRR(k=40),
        LinkPredNDCG(k=80),
    ]

    row = torch.randint(0, num_src_nodes, (num_edges, ))
    col = torch.randint(0, num_dst_nodes, (num_edges, ))
    edge_label_index = torch.stack([row, col], dim=0)

    pred = torch.rand(num_src_nodes, num_dst_nodes)
    pred[row, col] += 0.3  # Offset positive links by a little.
    pred_index_mat = pred.argsort(dim=1)

    metric_collection = LinkPredMetricCollection(metrics)
    assert str(metric_collection) == (
        'LinkPredMetricCollection([\n'
        '  LinkPredMAP@10: LinkPredMAP(k=10),\n'
        '  LinkPredPrecision@100: LinkPredPrecision(k=100),\n'
        '  LinkPredRecall@50: LinkPredRecall(k=50),\n'
        '  LinkPredF1@20: LinkPredF1(k=20),\n'
        '  LinkPredMRR@40: LinkPredMRR(k=40),\n'
        '  LinkPredNDCG@80: LinkPredNDCG(k=80),\n'
        '])')
    assert metric_collection.max_k == 100

    expected = {}
    for metric in metrics:
        metric.update(pred_index_mat[:, :metric.k], edge_label_index)
        out = metric.compute()
        expected[f'{metric.__class__.__name__}@{metric.k}'] = out
        metric.reset()

    metric_collection.update(pred_index_mat, edge_label_index)
    assert metric_collection.compute() == expected
    metric_collection.reset()
