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
from torch_geometric.testing import withPackage


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
    MaskType.attributes,
]
edge_mask_types = [
    MaskType.object,
    None,
]

x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@withPackage('captum')
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('task_level', ['node', 'edge', 'graph'])
@pytest.mark.parametrize('index', [2])
def test_captum_explainer_multiclass_classification(
    node_mask_type,
    edge_mask_type,
    task_level,
    index,
):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level=task_level,
        return_type='probs',
    )

    model = GCN(model_config)

    explainer = Explainer(
        model,
        algorithm=CaptumExplainer('IntegratedGradients',
                                  internal_batch_size=1),
        explanation_type='model',
        edge_mask_type=edge_mask_type,
        node_mask_type=node_mask_type,
        model_config=model_config,
    )

    explanation = explainer(
        x,
        edge_index,
        index=index,
        batch=batch,
        edge_label_index=edge_label_index,
    )

    if node_mask_type is not None:
        assert explanation.node_mask.size() == x.size()
    else:
        assert 'node_mask' not in explanation

    if edge_mask_type is not None:
        assert explanation.edge_mask.size() == (edge_index.size(1), )
    else:
        assert 'edge_mask' not in explanation
