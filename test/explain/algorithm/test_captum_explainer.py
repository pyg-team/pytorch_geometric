import pytest
import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer
from torch_geometric.explain.config import (
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

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        return x


# Test CaptumExplainer that is currently only returning None as masks
@pytest.mark.parametrize("node_mask_type", [
    None,
    MaskType.attributes,
    MaskType.object,
    MaskType.common_attributes,
])
@pytest.mark.parametrize("edge_mask_type",
                         [None, MaskType.attributes, MaskType.object])
@pytest.mark.parametrize("task_level",
                         [ModelTaskLevel.graph, ModelTaskLevel.node])
def test_captum_explainer(node_mask_type, edge_mask_type, task_level):
    # Initialize data, model and explainer
    model_config = ModelConfig(
        mode=ModelMode.multiclass_classification,
        task_level=task_level,
        return_type=ModelReturnType.probs,
    )
    model = GCN(model_config)

    allowed_node_mask_types = [MaskType.attributes, None]
    allowed_edge_mask_types = [MaskType.object, None]

    algorithm = CaptumExplainer(attribution_method="IntegratedGradients")

    if (node_mask_type in allowed_node_mask_types) and (
            edge_mask_type
            in allowed_edge_mask_types) and (node_mask_type != edge_mask_type):
        Explainer(
            model,
            algorithm=algorithm,
            explanation_type='model',
            model_config=model_config,
            edge_mask_type=edge_mask_type,
            node_mask_type=node_mask_type,
        )
    else:
        with pytest.raises(ValueError):
            Explainer(
                model,
                algorithm=algorithm,
                explanation_type='model',
                model_config=model_config,
                edge_mask_type=edge_mask_type,
                node_mask_type=node_mask_type,
            )
