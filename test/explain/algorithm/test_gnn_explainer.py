from typing import Optional, Union

import pytest
import torch
from torch import Tensor

from torch_geometric.explain import Explainer, Explanation, GNNExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode == ModelMode.multiclass_classification:
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        elif self.model_config.task_level == ModelTaskLevel.edge:
            assert edge_label_index is not None
            x = x[edge_label_index[0]] * x[edge_label_index[1]]

        if self.model_config.mode == ModelMode.binary_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.sigmoid()
        elif self.model_config.mode == ModelMode.multiclass_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.softmax(dim=-1)
            elif self.model_config.return_type == ModelReturnType.log_probs:
                x = x.log_softmax(dim=-1)

        return x


node_mask_types = [
    MaskType.object, MaskType.common_attributes, MaskType.attributes
]
edge_mask_types = [MaskType.object, None]
explanation_types = [ExplanationType.model, ExplanationType.phenomenon]
task_levels = [ModelTaskLevel.node, ModelTaskLevel.edge, ModelTaskLevel.graph]
indices = [None, 2, torch.arange(3)]

x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('explanation_type', explanation_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize('return_type',
                         [ModelReturnType.probs, ModelReturnType.raw])
@pytest.mark.parametrize('index', indices)
def test_gnn_explainer_binary_classification(edge_mask_type, node_mask_type,
                                             explanation_type, task_level,
                                             return_type, index,
                                             check_explanation):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level=task_level,
        return_type=return_type,
    )

    model = GCN(model_config)

    target = None
    if explanation_type == ExplanationType.phenomenon:
        with torch.no_grad():
            out = model(x, edge_index, batch, edge_label_index)
            if model_config.return_type == ModelReturnType.raw:
                target = (out > 0).long().view(-1)
            if model_config.return_type == ModelReturnType.probs:
                target = (out > 0.5).long().view(-1)

    explainer = create_explainer(model, explanation_type, model_config,
                                 node_mask_type, edge_mask_type)

    explanation = run_explainer(explainer, x, edge_index, batch,
                                edge_label_index, target, index)

    assert explainer.algorithm.node_mask is None
    assert explainer.algorithm.edge_mask is None

    check_explanation(explanation, node_mask_type, edge_mask_type)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('explanation_type', explanation_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize(
    'return_type',
    [ModelReturnType.log_probs, ModelReturnType.probs, ModelReturnType.raw])
@pytest.mark.parametrize('index', indices)
def test_gnn_explainer_multiclass_classification(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    return_type,
    index,
    check_explanation,
):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level=task_level,
        return_type=return_type,
    )

    model = GCN(model_config)

    target = None
    if explanation_type == ExplanationType.phenomenon:
        with torch.no_grad():
            target = model(x, edge_index, batch, edge_label_index).argmax(-1)

    explainer = create_explainer(model, explanation_type, model_config,
                                 node_mask_type, edge_mask_type)

    explanation = run_explainer(explainer, x, edge_index, batch,
                                edge_label_index, target, index)

    assert explainer.algorithm.node_mask is None
    assert explainer.algorithm.edge_mask is None

    check_explanation(explanation, node_mask_type, edge_mask_type)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('explanation_type', explanation_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize('index', indices)
def test_gnn_explainer_regression(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    index,
    check_explanation,
):
    model_config = ModelConfig(
        mode='regression',
        task_level=task_level,
    )

    model = GCN(model_config)

    target = None
    if explanation_type == ExplanationType.phenomenon:
        with torch.no_grad():
            target = model(x, edge_index, batch, edge_label_index)

    explainer = create_explainer(model, explanation_type, model_config,
                                 node_mask_type, edge_mask_type)

    explanation = run_explainer(explainer, x, edge_index, batch,
                                edge_label_index, target, index)

    assert explainer.algorithm.node_mask is None
    assert explainer.algorithm.edge_mask is None

    check_explanation(explanation, node_mask_type, edge_mask_type)


def create_explainer(model: torch.nn.Module, explanation_type: ExplanationType,
                     model_config: ModelConfig,
                     node_mask_type: Optional[MaskType],
                     edge_mask_type: Optional[MaskType]) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=2, edge_reduction='mean', edge_size=0.1),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )


def run_explainer(explainer: Explainer, x: Tensor, edge_index: Tensor,
                  batch: Tensor, edge_label_index: Tensor,
                  target: Optional[Tensor],
                  index: Optional[Union[int, Tensor]]) -> Explanation:
    return explainer(
        x,
        edge_index,
        target=target,
        index=index,
        batch=batch,
        edge_label_index=edge_label_index,
    )
