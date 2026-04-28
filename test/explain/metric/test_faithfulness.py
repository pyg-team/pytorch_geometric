import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    ModelConfig,
    unfaithfulness,
)
from torch_geometric.testing import get_random_edge_index


class DummyModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    def forward(self, x, edge_index):
        if self.model_config.return_type.value == 'probs':
            x = x.softmax(dim=-1)
        elif self.model_config.return_type.value == 'log_probs':
            x = x.log_softmax(dim=-1)
        return x


class HeteroDummyModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    def forward(self, x_dict, edge_index_dict, **kwargs):
        x = x_dict['paper']
        if self.model_config.return_type.value == 'probs':
            x = x.softmax(dim=-1)
        elif self.model_config.return_type.value == 'log_probs':
            x = x.log_softmax(dim=-1)
        return x


@pytest.mark.parametrize('top_k', [None, 2])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('node_mask_type', ['common_attributes', 'attributes'])
@pytest.mark.parametrize('return_type', ['raw', 'probs', 'log_probs'])
def test_unfaithfulness(top_k, explanation_type, node_mask_type, return_type):
    x = torch.randn(8, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ])

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type=return_type,
    )

    explainer = Explainer(
        DummyModel(model_config),
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type='object',
        model_config=model_config,
    )

    target = None
    if explanation_type == 'phenomenon':
        target = torch.randint(0, x.size(1), (x.size(0), ))

    explanation = explainer(x, edge_index, target=target,
                            index=torch.arange(4))

    metric = unfaithfulness(explainer, explanation, top_k)
    assert metric >= 0. and metric <= 1.


@pytest.mark.parametrize('top_k', [None, 2])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('node_mask_type', ['common_attributes', 'attributes'])
@pytest.mark.parametrize('return_type', ['raw', 'probs', 'log_probs'])
def test_unfaithfulness_hetero(top_k, explanation_type, node_mask_type,
                               return_type):
    data = HeteroData()
    data['paper'].x = torch.randn(8, 4)
    data['author'].x = torch.randn(10, 4)
    data['paper', 'paper'].edge_index = get_random_edge_index(8, 8, 10)
    data['paper', 'author'].edge_index = get_random_edge_index(8, 10, 10)
    data['author', 'paper'].edge_index = get_random_edge_index(10, 8, 10)

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type=return_type,
    )

    explainer = Explainer(
        HeteroDummyModel(model_config),
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type='object',
        model_config=model_config,
    )

    target = None
    if explanation_type == 'phenomenon':
        target = torch.randint(0, data['paper'].x.size(1),
                               (data['paper'].x.size(0), ))

    explanation = explainer(
        data.x_dict,
        data.edge_index_dict,
        target=target,
        index=torch.arange(4),
    )

    metric = unfaithfulness(explainer, explanation, top_k)
    assert metric >= 0. and metric <= 1.
