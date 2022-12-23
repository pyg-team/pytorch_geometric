import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    HeteroExplanation,
)
from torch_geometric.explain.config import ExplanationType


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.fixture
def data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    return data


class DummyModel(torch.nn.Module):
    def forward(self, x_dict, edge_index_dict, *args) -> torch.Tensor:
        return x_dict['paper'].mean().view(-1)


def test_get_prediction(data):
    model = DummyModel()
    assert model.training

    explainer = Explainer(
        model,
        algorithm=DummyExplainer(),
        explanation_type='phenomenon',
        node_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )
    pred = explainer.get_prediction(data.x_dict, data.edge_index_dict)
    assert model.training
    assert pred.size() == (1, )


@pytest.mark.parametrize('target', [None, torch.randn(2)])
@pytest.mark.parametrize('explanation_type', [x for x in ExplanationType])
def test_forward(data, target, explanation_type):
    model = DummyModel()

    explainer = Explainer(
        model,
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type='attributes',
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )

    if target is None and explanation_type == ExplanationType.phenomenon:
        with pytest.raises(ValueError):
            explainer(data.x_dict, data.edge_index_dict, target=target)
    else:
        explanation = explainer(
            data.x_dict,
            data.edge_index_dict,
            target=target
            if explanation_type == ExplanationType.phenomenon else None,
        )
        assert model.training
        assert isinstance(explanation, HeteroExplanation)
        assert 'node_mask' in explanation.available_explanations
        for key in explanation.node_types:
            assert explanation[key].node_mask.size() == data[key].x.size()


@pytest.mark.parametrize('threshold_value', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_hard_threshold(data, threshold_value, node_mask_type):

    explainer = Explainer(
        DummyModel(),
        algorithm=DummyExplainer(),
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
        threshold_config=('hard', threshold_value),
    )
    explanation = explainer(data.x_dict, data.edge_index_dict)
    assert 'node_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        for mask in explanation.collect(key).values():
            assert set(mask.unique().tolist()).issubset({0, 1})


@pytest.mark.parametrize('threshold_value', [1, 5, 10])
@pytest.mark.parametrize('threshold_type', ['topk', 'topk_hard'])
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_topk_threshold(data, threshold_value, threshold_type, node_mask_type):
    explainer = Explainer(
        DummyModel(),
        algorithm=DummyExplainer(),
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
        threshold_config=(threshold_type, threshold_value),
    )
    explanation = explainer(data.x_dict, data.edge_index_dict)

    assert 'node_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        for mask in explanation.collect(key).values():
            if threshold_type == 'topk':
                assert (mask > 0).sum() == min(mask.numel(), threshold_value)
                assert ((mask == 0).sum() == mask.numel() -
                        min(mask.numel(), threshold_value))
            else:
                assert (mask == 1).sum() == min(mask.numel(), threshold_value)
                assert ((mask == 0).sum() == mask.numel() -
                        min(mask.numel(), threshold_value))
