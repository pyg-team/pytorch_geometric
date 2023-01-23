import pytest
import torch

from torch_geometric.contrib.explain import GraphMaskExplainer
from torch_geometric.explain import Explainer
from torch_geometric.explain.config import (
    ModelConfig,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        assert model_config.mode == ModelMode.binary_classification

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        elif self.model_config.task_level == ModelTaskLevel.edge:
            assert edge_label_index is not None
            x = x[edge_label_index[0]] * x[edge_label_index[1]]

        return x


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize('task_level', ['node', 'edge', 'graph'])
def test_graph_mask_explainer(task_level):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level=task_level,
        return_type='raw',
    )

    model = GCN(model_config)

    explainer = Explainer(
        model=model,
        algorithm=GraphMaskExplainer(2, epochs=5),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config,
    )

    explanation = explainer(
        x,
        edge_index,
        batch=batch,
        edge_label_index=edge_label_index,
    )

    assert explanation.node_mask.size() == explanation.x.size()
    assert explanation.node_mask.min() >= 0
    assert explanation.node_mask.max() <= 1

    assert explanation.edge_mask.size() == (explanation.num_edges, )
    assert explanation.edge_mask.min() >= 0
    assert explanation.edge_mask.max() <= 1
