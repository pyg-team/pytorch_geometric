import pytest
import torch

from torch_geometric.contrib.explain import PGMExplainer
from torch_geometric.explain import Explainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.testing import withPackage


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode.value == 'multiclass_classification':
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_weight=None, batch=None, **kwargs):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()

        if self.model_config.task_level.value == 'graph':
            x = global_add_pool(x, batch)

        if self.model_config.mode.value == 'binary_classification':
            x = x.sigmoid()
        elif self.model_config.mode.value == 'multiclass_classification':
            x = x.log_softmax(dim=-1)

        return x


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
target = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@withPackage('pgmpy')
@withPackage('pandas')
@pytest.mark.parametrize('node_idx', [2, 6])
@pytest.mark.parametrize('task_level, perturbation_mode', [
    ('node', 'randint'),
    ('graph', 'mean'),
    ('graph', 'max'),
    ('graph', 'min'),
    ('graph', 'zero'),
])
def test_pgm_explainer_classification(node_idx, task_level, perturbation_mode):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level=task_level,
        return_type='raw',
    )

    model = GCN(model_config)
    logits = model(x, edge_index)
    target = logits.argmax(dim=1)

    explainer = Explainer(
        model=model,
        algorithm=PGMExplainer(feature_index=[0],
                               perturbation_mode=perturbation_mode),
        explanation_type='phenomenon',
        node_mask_type='object',
        model_config=model_config,
    )

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        index=node_idx,
        target=target,
    )

    assert 'node_mask' in explanation
    assert 'pgm_stats' in explanation
    assert explanation.node_mask.size(0) == explanation.num_nodes
    assert explanation.node_mask.min() >= 0
    assert explanation.node_mask.max() <= 1
