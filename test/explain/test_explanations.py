import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation


@pytest.fixture
def data():
    return Data(
        x=torch.randn(10, 5),
        edge_index=torch.randint(0, 10, (2, 20)),
        edge_attr=torch.randn(20, 3),
    )


def create_random_explanation(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.rand(data.x.size(0)) if node_mask else None
    edge_mask = torch.rand(data.edge_index.size(1)) if edge_mask else None
    node_feat_mask = torch.rand_like(data.x) if node_feat_mask else None
    edge_feat_mask = (torch.rand_like(data.edge_attr)
                      if edge_feat_mask else None)

    return Explanation(  # Create explanation.
        node_mask=node_mask, edge_mask=edge_mask,
        node_feat_mask=node_feat_mask, edge_feat_mask=edge_feat_mask, x=data.x,
        edge_index=data.edge_index, edge_attr=data.edge_attr)


@pytest.mark.parametrize('node_mask', [True, False])
@pytest.mark.parametrize('edge_mask', [True, False])
@pytest.mark.parametrize('node_feat_mask', [True, False])
@pytest.mark.parametrize('edge_feat_mask', [True, False])
def test_available_explanations(data, node_mask, edge_mask, node_feat_mask,
                                edge_feat_mask):
    expected = []
    if node_mask:
        expected.append('node_mask')
    if edge_mask:
        expected.append('edge_mask')
    if node_feat_mask:
        expected.append('node_feat_mask')
    if edge_feat_mask:
        expected.append('edge_feat_mask')

    explanation = create_random_explanation(
        data,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
    )

    assert set(explanation.available_explanations) == set(expected)


def test_get_explanation_data():
    pass


def test_validate_explanation(data):
    # validate random explanation
    explanation = create_random_explanation(data)
    explanation.validate(raise_on_error=True)

    # missmatch between x and node_mask shape
    with pytest.raises(ValueError):
        explanation = create_random_explanation(data)
        explanation.x = torch.randn(9, 5)
        explanation.validate(raise_on_error=True)

    # missmatch between x and node_feat_mask shape
    with pytest.raises(ValueError):
        explanation = create_random_explanation(data)
        explanation.x = torch.randn(10, 4)
        explanation.validate(raise_on_error=True)

    # missmatch between edge_index and edge_mask shape
    with pytest.raises(ValueError):
        explanation = create_random_explanation(data)
        explanation.edge_index = torch.randint(0, 10, (2, 21))
        explanation.validate(raise_on_error=True)

    # missmatch between edge_index and edge_mask shape
    with pytest.raises(ValueError):
        explanation = create_random_explanation(data)
        explanation.edge_attr = torch.randn(20, 1)
        explanation.validate(raise_on_error=True)

    # empty edge_attr with edge_mask present
    with pytest.raises(ValueError):
        explanation = create_random_explanation(data)
        explanation.edge_attr = None
        explanation.validate(raise_on_error=True)
