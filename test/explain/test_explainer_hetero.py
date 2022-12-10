import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    HeteroExplanation,
)
from torch_geometric.explain.config import ExplanationType
from torch_geometric.nn import SAGEConv, to_hetero


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


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

    def forward(self, x_dict, edge_index_dict, *args) -> torch.Tensor:
        return self.graph_sage(x_dict, edge_index_dict)['paper']


def test_get_prediction(data):
    model = HeteroSAGE(data.metadata())
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

    assert pred.size() == (8, 1)


@pytest.mark.parametrize('target', [None, torch.randn(2)])
@pytest.mark.parametrize('explanation_type', [x for x in ExplanationType])
def test_forward(data, target, explanation_type):
    model = HeteroSAGE(data.metadata())

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
        explanation = explainer(data.x_dict, data.edge_index_dict,
                                target=target)
        assert isinstance(explanation, HeteroExplanation)
        assert 'node_feat_mask' in explanation.available_explanations
        for node_type, node_feat_mask in explanation.node_feat_mask.items():
            assert node_feat_mask.size() == data[node_type].x.size()


@pytest.mark.parametrize('threshold_value', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_hard_threshold(data, threshold_value, node_mask_type):
    model = HeteroSAGE(data.metadata())

    explainer = Explainer(
        model,
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

    if node_mask_type == 'object':
        assert 'node_mask' in explanation.available_explanations
    elif node_mask_type == 'attributes':
        assert 'node_feat_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        masks = explanation[key]
        for mask in masks.values():
            assert set(mask.unique().tolist()).issubset({0, 1})


@pytest.mark.parametrize('threshold_value', list(range(1, 10)))
@pytest.mark.parametrize('threshold_type', ['topk', 'topk_hard'])
@pytest.mark.parametrize('node_mask_type', ['object', 'attributes'])
def test_topk_threshold(data, threshold_value, threshold_type, node_mask_type):
    model = HeteroSAGE(data.metadata())

    explainer = Explainer(
        model,
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

    if node_mask_type == 'object':
        assert 'node_mask' in explanation.available_explanations
    elif node_mask_type == 'attributes':
        assert 'node_feat_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations

    for key in explanation.available_explanations:
        masks = explanation[key]
        for mask in masks.values():
            if threshold_type == 'topk':
                assert (mask > 0).sum() == min(mask.numel(), threshold_value)
                assert ((mask == 0).sum() == mask.numel() -
                        min(mask.numel(), threshold_value))
            else:
                assert (mask == 1).sum() == min(mask.numel(), threshold_value)
                assert ((mask == 0).sum() == mask.numel() -
                        min(mask.numel(), threshold_value))
