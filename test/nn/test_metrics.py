from typing import List

import pytest
import torch

from torch_geometric.nn.metrics import LinkPredPrecision


@pytest.mark.parametrize('num_src_nodes', [100])
@pytest.mark.parametrize('num_dst_nodes', [1000])
@pytest.mark.parametrize('num_edges', [3000])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('top_k', [1, 10, 100])
def test_metrics(num_src_nodes, num_dst_nodes, num_edges, batch_size, top_k):
    row = torch.randint(0, num_src_nodes, (num_edges, ))
    col = torch.randint(0, num_dst_nodes, (num_edges, ))
    edge_label_index = torch.stack([row, col], dim=0)

    pred = torch.rand(num_src_nodes, num_dst_nodes)
    pred[row, col] += 0.3  # Offset positive links by a little.
    top_k_pred_mat = pred.topk(top_k, dim=1)[1]

    metric = LinkPredPrecision(top_k)
    assert str(metric) == f'LinkPredPrecision({top_k})'

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
            precision = float(mask.sum() / top_k)
            values.append(precision)
    expected = torch.tensor(values).mean()
    assert torch.allclose(out, expected)
