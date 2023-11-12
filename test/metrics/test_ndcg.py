import pytest
import torch

from torch_geometric.metrics import NormalizedDCG


def test_ndcg_computes_metric():
    metric = NormalizedDCG(k=2)

    pred_mat = torch.Tensor([[1, 0, 2], [1, 2, 0], [0, 2, 1]])

    edge_label_index = torch.LongTensor(
        [[0, 0, 2, 2], [0, 1, 2, 0]], ).contiguous()

    metric.update(pred_mat, edge_label_index)

    result = metric.compute()

    assert pytest.approx(result.item(), 0.0001) == 0.386853


def test_ndcg_without_k():
    metric = NormalizedDCG()

    pred_mat = torch.Tensor([[1, 0, 2], [1, 2, 0], [0, 2, 1]])

    edge_label_index = torch.LongTensor(
        [[0, 0, 2, 2], [0, 1, 2, 0]], ).contiguous()

    metric.update(pred_mat, edge_label_index)

    result = metric.compute()

    assert pytest.approx(result.item(), 0.0001) == 0.6934


def test_ndcg_multiple_updates():
    metric = NormalizedDCG()

    pred_mat_a = torch.Tensor([[1, 0, 2], [1, 2, 0], [0, 2, 1]])

    pred_mat_b = torch.Tensor([[0, 1, 0], [1, 1, 0], [1, 1, 1]])

    edge_label_index = torch.LongTensor(
        [[0, 0, 2, 2], [0, 1, 2, 0]], ).contiguous()

    metric.update(pred_mat_a, edge_label_index)
    metric.update(pred_mat_b, edge_label_index)

    result = metric.compute()

    assert pytest.approx(result.item(), 0.0001) == 0.8266
