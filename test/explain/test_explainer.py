import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import DummyExplainer, Explainer
from torch_geometric.explain.config import ExplanationType


@pytest.fixture
def data():
    return Data(
        x=torch.randn(8, 3, requires_grad=True),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                                 [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]),
        edge_attr=torch.randn(14, 3),
    )


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, edge_index):
        return x.mean().repeat(self.out_dim)


@pytest.mark.parametrize('model', [
    DummyModel(out_dim=1),
    DummyModel(out_dim=2),
])
def test_get_prediction(data, model):
    explainer = Explainer(
        model,
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='phenomenon',
            node_mask_type='object',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )
    pred = explainer.get_prediction(data.x, data.edge_index)
    assert pred.size() == (model.out_dim, )


@pytest.mark.parametrize('target', [None, torch.randn(2)])
@pytest.mark.parametrize('explanation_type', [x for x in ExplanationType])
def test_forward(data, target, explanation_type):
    explainer = Explainer(
        DummyModel(out_dim=2),
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type=explanation_type,
            node_mask_type='attributes',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )

    if target is None and explanation_type == ExplanationType.phenomenon:
        with pytest.raises(ValueError):
            explainer(data.x, data.edge_index, target=target)
    else:
        explanation = explainer(data.x, data.edge_index, target=target)
        assert 'node_feat_mask' in explanation.available_explanations
        assert explanation.node_feat_mask.size() == data.x.size()


@pytest.mark.parametrize('threshold_value', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_hard_threshold(data, threshold_value, node_mask_type):
    explainer = Explainer(
        DummyModel(out_dim=2),
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type='object',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
        threshold_config=('hard', threshold_value),
    )
    explanation = explainer(data.x, data.edge_index)
    if node_mask_type == 'object':
        assert 'node_mask' in explanation.available_explanations
    elif node_mask_type == 'attributes':
        assert 'node_feat_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        mask = explanation[key]
        assert set(mask.unique().tolist()).issubset({0, 1})


@pytest.mark.parametrize('threshold_value', list(range(1, 10)))
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_topk_threshold(data, threshold_value, node_mask_type):
    explainer = Explainer(
        DummyModel(out_dim=2),
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type='object',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
        threshold_config=('topk', threshold_value),
    )
    explanation = explainer(data.x, data.edge_index)

    if node_mask_type == 'object':
        assert 'node_mask' in explanation.available_explanations
    elif node_mask_type == 'attributes':
        assert 'node_feat_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        mask = explanation[key]
        assert (mask > 0).sum() == min(mask.numel(), threshold_value)
        assert ((mask == 0).sum() == mask.numel() -
                min(mask.numel(), threshold_value))


@pytest.mark.parametrize('threshold_value', list(range(1, 10)))
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_topk_hard_threshold(data, threshold_value, node_mask_type):
    explainer = Explainer(
        DummyModel(out_dim=2),
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type='object',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
        threshold_config=('topk_hard', threshold_value),
    )
    explanation = explainer(data.x, data.edge_index)

    if node_mask_type == 'object':
        assert 'node_mask' in explanation.available_explanations
    elif node_mask_type == 'attributes':
        assert 'node_feat_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        mask = explanation[key]
        assert (mask == 1).sum() == min(mask.numel(), threshold_value)
        assert ((mask == 0).sum() == mask.numel() -
                min(mask.numel(), threshold_value))
