from typing import Optional, Union

import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation
from torch_geometric.explain.config import MaskType


@pytest.fixture
def data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        [4, 5, 6, 4, 5, 6, 4, 5, 6, 7],
    ])
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['paper', 'author'].edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        [4, 5, 6, 4, 5, 6, 4, 5, 6, 7],
    ])
    data['paper', 'author'].edge_attr = torch.randn(10, 8)
    return data


def create_random_explanation(
    data: HeteroData,
    node_mask_type: Optional[Union[MaskType, str]] = None,
    edge_mask_type: Optional[Union[MaskType, str]] = None,
):
    if node_mask_type is not None:
        node_mask_type = MaskType(node_mask_type)
    if edge_mask_type is not None:
        edge_mask_type = MaskType(edge_mask_type)

    out = HeteroExplanation()

    for key in ['paper', 'author']:
        out[key].x = data[key].x
        if node_mask_type == MaskType.object:
            out[key].node_mask = torch.rand(data[key].num_nodes, 1)
        elif node_mask_type == MaskType.common_attributes:
            out[key].node_mask = torch.rand(1, data[key].num_features)
        elif node_mask_type == MaskType.attributes:
            out[key].node_mask = torch.rand_like(data[key].x)

    for key in [('paper', 'paper'), ('paper', 'author')]:
        out[key].edge_index = data[key].edge_index
        out[key].edge_attr = data[key].edge_attr
        if edge_mask_type == MaskType.object:
            out[key].edge_mask = torch.rand(data[key].num_edges)

    return out


@pytest.mark.parametrize('node_mask_type',
                         [None, 'object', 'common_attributes', 'attributes'])
@pytest.mark.parametrize('edge_mask_type', [None, 'object'])
def test_available_explanations(data, node_mask_type, edge_mask_type):
    expected = []
    if node_mask_type:
        expected.append('node_mask')
    if edge_mask_type:
        expected.append('edge_mask')

    explanation = create_random_explanation(
        data,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )

    assert set(explanation.available_explanations) == set(expected)


def test_validate_explanation(data):
    explanation = create_random_explanation(data)
    explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 8 nodes"):
        explanation = create_random_explanation(data)
        explanation['paper'].node_mask = torch.rand(5, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 5 features"):
        explanation = create_random_explanation(data, 'attributes')
        explanation['paper'].x = torch.randn(8, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 10 edges"):
        explanation = create_random_explanation(data)
        explanation['paper', 'paper'].edge_mask = torch.randn(5)
        explanation.validate(raise_on_error=True)


def test_node_mask(data):
    explanation = HeteroExplanation()
    explanation['paper'].node_mask = torch.tensor([[1.], [0.], [1.], [1.]])
    explanation['author'].node_mask = torch.tensor([[1.], [0.], [1.], [1.]])
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out['paper'].node_mask.size() == (3, 1)
    assert out['author'].node_mask.size() == (3, 1)

    out = explanation.get_complement_subgraph()
    assert out['paper'].node_mask.size() == (1, 1)
    assert out['author'].node_mask.size() == (1, 1)


def test_edge_mask(data):
    explanation = HeteroExplanation()
    explanation['paper'].num_nodes = 4
    explanation['author'].num_nodes = 4
    explanation['paper', 'author'].edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    explanation['paper', 'author'].edge_mask = torch.tensor([1., 0., 1., 1.])

    out = explanation.get_explanation_subgraph()
    assert out['paper'].num_nodes == 4
    assert out['author'].num_nodes == 4
    assert out['paper', 'author'].edge_mask.size() == (3, )
    assert torch.equal(out['paper', 'author'].edge_index,
                       torch.tensor([[0, 2, 3], [0, 2, 3]]))

    out = explanation.get_complement_subgraph()
    assert out['paper'].num_nodes == 4
    assert out['author'].num_nodes == 4
    assert out['paper', 'author'].edge_mask.size() == (1, )
    assert torch.equal(out['paper', 'author'].edge_index,
                       torch.tensor([[1], [1]]))
