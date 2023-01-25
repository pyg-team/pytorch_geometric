import torch

from torch_geometric.explain import groundtruth_metrics


def test_groundtruth_metrics():
    pred_mask = torch.rand(10)
    target_mask = torch.rand(10)

    accuracy, recall, precision, f1_score, auroc = groundtruth_metrics(
        pred_mask, target_mask)

    assert accuracy >= 0.0 and accuracy <= 1.0
    assert recall >= 0.0 and recall <= 1.0
    assert precision >= 0.0 and precision <= 1.0
    assert f1_score >= 0.0 and f1_score <= 1.0
    assert auroc >= 0.0 and auroc <= 1.0


def test_perfect_groundtruth_metrics():
    pred_mask = target_mask = torch.rand(10)

    accuracy, recall, precision, f1_score, auroc = groundtruth_metrics(
        pred_mask, target_mask)

    assert round(accuracy, 6) == 1.0
    assert round(recall, 6) == 1.0
    assert round(precision, 6) == 1.0
    assert round(f1_score, 6) == 1.0
    assert round(auroc, 6) == 1.0


def test_groundtruth_true_negative():
    pred_mask = target_mask = torch.zeros(10)

    accuracy, recall, precision, f1_score, auroc = groundtruth_metrics(
        pred_mask, target_mask)

    assert round(accuracy, 6) == 1.0
    assert round(recall, 6) == 0.0
    assert round(precision, 6) == 0.0
    assert round(f1_score, 6) == 0.0
    assert round(auroc, 6) == 0.0
