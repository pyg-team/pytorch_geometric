from typing import List

import pytest
import torch

from torch_geometric.nn.metrics import (
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
    top_k_pred_mat = pred.topk(k, dim=1)[1]

    metric = LinkPredPrecision(k)
    assert str(metric) == f'LinkPredPrecision(k={k})'

    for node_id in torch.split(torch.randperm(num_src_nodes), batch_size):
        mask = torch.isin(edge_label_index[0], node_id)

        y_batch, y_index = edge_label_index[:, mask]
        # Remap `y_batch` back to `[0, batch_size - 1]` range:
        arange = torch.empty(num_src_nodes, dtype=node_id.dtype)
        arange[node_id] = torch.arange(node_id.numel())
        y_batch = arange[y_batch]

        metric.update(top_k_pred_mat[node_id], (y_batch, y_index))

    out = metric.compute()
    metric.reset()

    values: List[float] = []
    for i in range(num_src_nodes):  # Naive computation per node:
        y_index = col[row == i]
        if y_index.numel() > 0:
            mask = torch.isin(top_k_pred_mat[i], y_index)
            precision = float(mask.sum() / k)
            values.append(precision)
    expected = torch.tensor(values).mean()
    assert torch.allclose(out, expected)


def test_recall():
    pred_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 2, 1]])

    metric = LinkPredRecall(k=2)
    assert str(metric) == f'LinkPredRecall(k={2})'
    metric.update(pred_mat, edge_label_index)
    result = metric.compute()

    assert float(result) == pytest.approx(0.5 * (2 / 3 + 0.5))


def test_ndcg():
    pred_mat = torch.tensor([[1, 0], [1, 2], [0, 2]])
    edge_label_index = torch.tensor([[0, 0, 2, 2], [0, 1, 2, 1]])

    metric = LinkPredNDCG(k=2)
    assert str(metric) == f'LinkPredNDCG(k={2})'
    metric.update(pred_mat, edge_label_index)
    result = metric.compute()

    assert float(result) == pytest.approx(0.6934264)
