import os.path
import random
import sys
from typing import Optional, Union

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import MaskType
from torch_geometric.testing import withPackage


def create_random_explanation(
    data: Data,
    node_mask_type: Optional[Union[MaskType, str]] = None,
    edge_mask_type: Optional[Union[MaskType, str]] = None,
):
    if node_mask_type is not None:
        node_mask_type = MaskType(node_mask_type)
    if edge_mask_type is not None:
        edge_mask_type = MaskType(edge_mask_type)

    if node_mask_type == MaskType.object:
        node_mask = torch.rand(data.x.size(0), 1)
    elif node_mask_type == MaskType.common_attributes:
        node_mask = torch.rand(1, data.x.size(1))
    elif node_mask_type == MaskType.attributes:
        node_mask = torch.rand_like(data.x)
    else:
        node_mask = None

    if edge_mask_type == MaskType.object:
        edge_mask = torch.rand(data.edge_index.size(1))
    else:
        edge_mask = None

    return Explanation(  # Create explanation.
        node_mask=node_mask,
        edge_mask=edge_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


@pytest.mark.parametrize('node_mask_type',
                         [None, 'object', 'common_attributes', 'attributes'])
@pytest.mark.parametrize('edge_mask_type', [None, 'object'])
def test_available_explanations(data, node_mask_type, edge_mask_type):
    expected = []
    if node_mask_type is not None:
        expected.append('node_mask')
    if edge_mask_type is not None:
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

    with pytest.raises(ValueError, match="with 5 nodes"):
        explanation = create_random_explanation(data, node_mask_type='object')
        explanation.x = torch.randn(5, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 4 features"):
        explanation = create_random_explanation(data, 'attributes')
        explanation.x = torch.randn(4, 4)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 7 edges"):
        explanation = create_random_explanation(data, edge_mask_type='object')
        explanation.edge_index = torch.randint(0, 4, (2, 7))
        explanation.validate(raise_on_error=True)


def test_node_mask(data):
    node_mask = torch.tensor([[1.], [0.], [1.], [0.]])

    explanation = Explanation(
        node_mask=node_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.node_mask.size() == (2, 1)
    assert (out.node_mask > 0.0).sum() == 2
    assert out.x.size() == (2, 3)
    assert out.edge_index.size(1) <= 6
    assert out.edge_index.size(1) == out.edge_attr.size(0)

    out = explanation.get_complement_subgraph()
    assert out.node_mask.size() == (2, 1)
    assert (out.node_mask == 0.0).sum() == 2
    assert out.x.size() == (2, 3)
    assert out.edge_index.size(1) <= 6
    assert out.edge_index.size(1) == out.edge_attr.size(0)


def test_edge_mask(data):
    edge_mask = torch.tensor([1., 0., 1., 0., 0., 1.])

    explanation = Explanation(
        edge_mask=edge_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.x.size() == (4, 3)
    assert out.edge_mask.size() == (3, )
    assert (out.edge_mask > 0.0).sum() == 3
    assert out.edge_index.size() == (2, 3)
    assert out.edge_attr.size() == (3, 3)

    out = explanation.get_complement_subgraph()
    assert out.x.size() == (4, 3)
    assert out.edge_mask.size() == (3, )
    assert (out.edge_mask == 0.0).sum() == 3
    assert out.edge_index.size() == (2, 3)
    assert out.edge_attr.size() == (3, 3)


@withPackage('matplotlib', 'pandas')
@pytest.mark.parametrize('top_k', [2, None])
@pytest.mark.parametrize('node_mask_type', [None, 'attributes'])
def test_visualize_feature_importance(data, top_k, node_mask_type):
    explanation = create_random_explanation(data, node_mask_type)

    path = os.path.join('/', 'tmp', f'{random.randrange(sys.maxsize)}.png')

    if node_mask_type is None:
        with pytest.raises(ValueError, match="node_mask' is not"):
            explanation.visualize_feature_importance(path, top_k=top_k)
    else:
        assert not os.path.exists(path)
        explanation.visualize_feature_importance(path, top_k=top_k)
        assert os.path.exists(path)
        os.remove(path)
