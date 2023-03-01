from typing import Optional

import pytest
import torch

from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.testing import withCUDA


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig,
                 mode: ModelMode = ModelMode.multiclass_classification):
        super().__init__()
        self.model_config = model_config
        self.mode = mode

        self.conv1 = GCNConv(3, 16)
        if mode == ModelMode.multiclass_classification:
            self.conv2 = GCNConv(16, 7)
        elif mode == ModelMode.binary_classification:
            self.conv2 = GCNConv(16, 1)
        elif mode == ModelMode.regression:
            self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        return x


def create_model_config(
        task_level: ModelTaskLevel,
        mode: ModelMode = ModelMode.multiclass_classification) -> ModelConfig:
    return ModelConfig(
        mode=mode,
        task_level=task_level,
        return_type='raw',
    )


def create_explainer(model: torch.nn.Module, explanation_type: ExplanationType,
                     model_config: ModelConfig, device='cpu',
                     node_mask_type: Optional[MaskType] = None) -> Explainer:
    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type=explanation_type,
        edge_mask_type='object',
        node_mask_type=node_mask_type,
        model_config=model_config,
    )


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.regression,
    ModelMode.multiclass_classification,
    ModelMode.binary_classification,
])
def test_pg_explainer_node(device, check_explanation, mode):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)

    if mode == ModelMode.multiclass_classification:
        target = torch.randint(7, (x.size(0), ), device=device)
    elif mode == ModelMode.binary_classification:
        target = torch.randint(2, (x.size(0), ), device=device)
    else:
        target = torch.randn((x.size(0), ), device=device)

    model_config = create_model_config(task_level=ModelTaskLevel.node,
                                       mode=mode)

    model = GCN(model_config, mode=mode).to(device)

    explainer = create_explainer(model, ExplanationType.phenomenon,
                                 model_config, device=device)

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        for index in range(x.size(0)):
            loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                             target=target, index=index)
            assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target, index=0)

    check_explanation(explanation, None, explainer.edge_mask_type)


@withCUDA
def test_pg_explainer_graph(device, check_explanation):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)
    target = torch.randint(7, (1, ), device=device)

    model_config = create_model_config(task_level=ModelTaskLevel.graph)

    model = GCN(model_config).to(device)

    explainer = create_explainer(model, ExplanationType.phenomenon,
                                 model_config, device=device)

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(10):
        loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                         target=target)
        assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target)

    check_explanation(explanation, None, explainer.edge_mask_type)


def test_pg_explainer_supports():
    # Test unsupported model task level:
    model_config = create_model_config(task_level=ModelTaskLevel.edge)
    model = GCN(model_config)
    with pytest.raises(ValueError, match="not support the given explanation"):
        create_explainer(model, ExplanationType.phenomenon, model_config)

    # Test unsupported explanation type:
    model_config = create_model_config(task_level=ModelTaskLevel.node)
    model = GCN(model_config)
    with pytest.raises(ValueError, match="not support the given explanation"):
        create_explainer(model, ExplanationType.model, model_config)

    # Test unsupported node mask:
    with pytest.raises(ValueError, match="not support the given explanation"):
        create_explainer(model, ExplanationType.phenomenon, model_config,
                         node_mask_type=MaskType.object)
