import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation


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
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    out = HeteroExplanation()

    for key in ['paper', 'author']:
        out[key].x = data[key].x
        if node_mask:
            out[key].node_mask = torch.rand(out[key].num_nodes)
        if node_feat_mask:
            out[key].node_feat_mask = torch.rand_like(out[key].x)

    for key in [('paper', 'paper'), ('paper', 'author')]:
        out[key].edge_index = data[key].edge_index
        out[key].edge_attr = data[key].edge_attr
        if edge_mask:
            out[key].edge_mask = torch.rand(data[key].num_edges)
        if edge_feat_mask:
            out[key].edge_feat_mask = torch.rand_like(data[key].edge_attr)

    return out


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

    with pytest.raises(ValueError, match="with 8 nodes"):
        explanation = create_random_explanation(data)
        explanation['paper'].node_mask = torch.rand(5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 10 edges"):
        explanation = create_random_explanation(data)
        explanation['paper', 'paper'].edge_mask = torch.randn(5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match=r"of shape \[8, 16\]"):
        explanation = create_random_explanation(data)
        explanation['paper'].node_feat_mask = torch.rand(8, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match=r"of shape \[10, 16\]"):
        explanation = create_random_explanation(data)
        explanation['paper', 'paper'].edge_feat_mask = torch.rand(10, 5)
        explanation.validate(raise_on_error=True)


def test_node_mask(data):
    explanation = HeteroExplanation()
    explanation['paper'].node_mask = torch.tensor([1., 0., 1., 1.])
    explanation['author'].node_mask = torch.tensor([1., 0., 1., 1.])
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out['paper'].node_mask.size() == (3, )
    assert out['author'].node_mask.size() == (3, )

    out = explanation.get_complement_subgraph()
    assert out['paper'].node_mask.size() == (1, )
    assert out['author'].node_mask.size() == (1, )


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
