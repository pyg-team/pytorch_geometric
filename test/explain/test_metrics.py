import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.metrics import get_groundtruth_metrics


@pytest.fixture
def data1():
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ]),
        edge_attr=torch.randn(6, 3),
    )


@pytest.fixture
def data2():
    return Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([
            [0, 0, 0],
            [1, 2, 3],
        ]),
        edge_attr=torch.randn(3, 2),
    )


def create_groundtruth(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.Tensor([0.123, 0.617, 0.815, 0.900
                              ]) if node_mask else None
    edge_mask = torch.Tensor([0.243, 0.660, 0.780, 0.000, 0.000, 1.000
                              ]) if edge_mask else None
    node_feat_mask = torch.Tensor(
        [[0.000, 0.450, 1.000], [0.234, 0.100, 0.000], [1.000, 1.000, 0.000],
         [0.000, 0.000, 0.100]]) if node_feat_mask else None
    edge_feat_mask = torch.Tensor(
        [[0.600, 0.400, 0.000], [0.234, 0.100, 0.000], [1.000, 0.000, 0.000],
         [0.000, 0.000, 0.100]]) if edge_feat_mask else None

    return Explanation(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


def create_explanation(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.Tensor([0.245, 0.312, 0.670, 0.899
                              ]) if node_mask else None
    edge_mask = torch.Tensor([0.342, 0.660, 0.780, 0.000, 1.000, 0.000
                              ]) if edge_mask else None
    node_feat_mask = torch.Tensor(
        [[0.000, 0.450, 0.990], [0.234, 0.100, 0.000], [1.000, 1.000, 0.000],
         [0.000, 0.000, 0.100]]) if node_feat_mask else None
    edge_feat_mask = torch.Tensor(
        [[0.600, 0.400, 0.000], [0.234, 0.100, 0.000], [1.000, 0.500, 0.000],
         [0.100, 0.000, 0.100]]) if edge_feat_mask else None

    return Explanation(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


def create_groundtruth_with_only_tn(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.Tensor([0.000, 0.000, 0.000]) if node_mask else None
    edge_mask = torch.Tensor([0.000, 0.000, 0.000]) if edge_mask else None
    node_feat_mask = torch.Tensor([[0.000, 0.000], [0.000, 0.000],
                                   [0.000, 0.000]]) if node_feat_mask else None
    edge_feat_mask = torch.Tensor([[0.000, 0.000], [0.000, 0.000],
                                   [0.000, 0.000]]) if edge_feat_mask else None

    return Explanation(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


def create_explanation_with_only_tn(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.Tensor([0.000, 0.000, 0.000]) if node_mask else None
    edge_mask = torch.Tensor([0.000, 0.000, 0.000]) if edge_mask else None
    node_feat_mask = torch.Tensor([[0.000, 0.000], [0.000, 0.000],
                                   [0.000, 0.000]]) if node_feat_mask else None
    edge_feat_mask = torch.Tensor([[0.000, 0.000], [0.000, 0.000],
                                   [0.000, 0.000]]) if edge_feat_mask else None

    return Explanation(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


def test_accuracy_metrics(data1):
    ground_truth = create_groundtruth(
        data1,
        node_mask=True,
        edge_mask=True,
        node_feat_mask=True,
        edge_feat_mask=False,
    )
    explanation = create_explanation(
        data1,
        node_mask=True,
        edge_mask=True,
        node_feat_mask=True,
        edge_feat_mask=False,
    )
    accuracy_metrics = get_groundtruth_metrics(explanation, ground_truth)
    assert torch.round(accuracy_metrics[0], decimals=4) == torch.tensor(0.9091)
    assert torch.round(accuracy_metrics[1], decimals=4) == torch.tensor(0.9333)
    assert torch.round(accuracy_metrics[2], decimals=4) == torch.tensor(0.9333)
    assert torch.round(accuracy_metrics[3], decimals=4) == torch.tensor(0.9333)
    assert torch.round(accuracy_metrics[4], decimals=4) == torch.tensor(1.0)


def test_accuracy_metrics_when_precision_and_recall_is_zero(data2):
    ground_truth = create_groundtruth_with_only_tn(
        data2,
        node_mask=True,
        edge_mask=True,
        node_feat_mask=False,
        edge_feat_mask=False,
    )
    explanation = create_explanation_with_only_tn(
        data2,
        node_mask=True,
        edge_mask=True,
        node_feat_mask=False,
        edge_feat_mask=False,
    )
    accuracy_metrics = get_groundtruth_metrics(explanation, ground_truth)
    assert torch.round(accuracy_metrics[0], decimals=4) == torch.tensor(1.)
    assert torch.round(accuracy_metrics[1], decimals=4) == torch.tensor(0.)
    assert torch.round(accuracy_metrics[2], decimals=4) == torch.tensor(0.)
    assert torch.round(accuracy_metrics[3], decimals=4) == torch.tensor(0.)
    assert torch.round(accuracy_metrics[4], decimals=4) == torch.tensor(0.)
