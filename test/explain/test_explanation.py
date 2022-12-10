import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.testing import withPackage


@pytest.fixture
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ]),
        edge_attr=torch.randn(6, 3),
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
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


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


def test_validate_explanation(data):
    explanation = create_random_explanation(data)
    explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 5 nodes"):
        explanation = create_random_explanation(data)
        explanation.x = torch.randn(5, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match=r"of shape \[4, 4\]"):
        explanation = create_random_explanation(data)
        explanation.x = torch.randn(4, 4)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 7 edges"):
        explanation = create_random_explanation(data)
        explanation.edge_index = torch.randint(0, 4, (2, 7))
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match=r"of shape \[6, 4\]"):
        explanation = create_random_explanation(data)
        explanation.edge_attr = torch.randn(6, 4)
        explanation.validate(raise_on_error=True)


def test_node_mask(data):
    node_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    explanation = Explanation(
        node_mask=node_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.node_mask.size() == (2, )
    assert (out.node_mask > 0.0).sum() == 2
    assert out.x.size() == (2, 3)
    assert out.edge_index.size() == (2, 1)
    assert out.edge_attr.size() == (1, 3)

    out = explanation.get_complement_subgraph()
    assert out.node_mask.size() == (2, )
    assert (out.node_mask == 0.0).sum() == 2
    assert out.x.size() == (2, 3)
    assert out.edge_index.size() == (2, 1)
    assert out.edge_attr.size() == (1, 3)


def test_edge_mask(data):
    edge_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

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


@withPackage('matplotlib')
@pytest.mark.parametrize('top_k', [2, None])
@pytest.mark.parametrize('node_feat_mask', [True, False])
def test_visualize_feature_importance(data, top_k, node_feat_mask):
    explanation = create_random_explanation(data,
                                            node_feat_mask=node_feat_mask)
    num_feats_plotted = top_k if top_k is not None else data.x.shape[-1]
    if not node_feat_mask:
        with pytest.raises(ValueError, match="node_feat_mask' not available"):
            _ = explanation.visualize_feature_importance(top_k=top_k)
    else:
        ax = explanation.visualize_feature_importance(top_k=top_k)
        assert len(ax.yaxis.get_ticklabels()) == num_feats_plotted
