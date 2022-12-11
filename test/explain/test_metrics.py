import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.metrics import groundtruth_metrics


@pytest.fixture
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ]),
        edge_attr=torch.randn(6, 3),
    )


def create_random_explanation(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.rand(data.x.size(0)) if node_mask else None
    edge_mask = torch.rand(data.edge_index.size(1)) if edge_mask else None
    node_feat_mask = torch.rand_like(data.x) if node_feat_mask else None
    edge_feat_mask = (torch.rand_like(data.edge_attr)
                      if edge_feat_mask else None)

    return Explanation(  # Create explanation.
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


def test_masks(data):

    ex_node_mask = [True, False]
    ex_edge_mask = [True, False]
    ex_node_feat_mask = [True, False]
    ex_edge_feat_mask = [True, False]
    gt_node_mask = [True, False]
    gt_edge_mask = [True, False]
    gt_node_feat_mask = [True, False]
    gt_edge_feat_mask = [True, False]

    explanation = create_random_explanation(
        data,
        node_mask=ex_node_mask,
        edge_mask=ex_edge_mask,
        node_feat_mask=ex_node_feat_mask,
        edge_feat_mask=ex_edge_feat_mask,
    )

    groundtruth = create_random_explanation(
        data,
        node_mask=gt_node_mask,
        edge_mask=gt_edge_mask,
        node_feat_mask=gt_node_feat_mask,
        edge_feat_mask=gt_edge_feat_mask,
    )

    accuracy_metrics = groundtruth_metrics(explanation, groundtruth)
    assert accuracy_metrics[0] == 1.0
    assert accuracy_metrics[1] == 1.0
    assert accuracy_metrics[2] == 1.0
    assert accuracy_metrics[3] == 1.0
    assert accuracy_metrics[4] == 0.5
