import pytest
import torch

from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.explain.config import ModelConfig, ModelTaskLevel
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.testing import withCUDA


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


@withCUDA
def test_pg_explainer_node(device):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)
    target = torch.randint(7, (x.size(0), ), device=device)

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explanation = explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        for index in range(x.size(0)):
            loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                             target=target, index=index)
            assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target, index=0)

    assert 'node_mask' not in explanation
    assert explanation.edge_mask.size() == (explanation.num_edges, )
    assert explanation.edge_mask.min() >= 0
    assert explanation.edge_mask.max() <= 1


@withCUDA
def test_pg_explainer_graph(device):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)
    target = torch.randint(7, (1, ), device=device)

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='graph',
        return_type='raw',
    )

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=10).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explanation = explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(10):
        loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                         target=target)
        assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target)

    assert 'node_mask' not in explanation
    assert explanation.edge_mask.size() == (explanation.num_edges, )
    assert explanation.edge_mask.min() >= 0
    assert explanation.edge_mask.max() <= 1
