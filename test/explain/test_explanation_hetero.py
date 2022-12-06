import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation


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
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['author', 'paper'].edge_attr = torch.randn(10, 12)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    data['paper', 'author'].edge_attr = torch.randn(10, 8)
    return data


def create_random_explanation(
    data: HeteroData,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = {
        "paper": torch.rand(data["paper"].x.size(0)),
        "author": torch.rand(data["author"].x.size(0)),
    } if node_mask else None

    edge_mask = {
        ("paper", "to", "paper"):
        torch.rand(data[("paper", "to", "paper")].edge_index.size(1)),
        ("author", "to", "paper"):
        torch.rand(data[("author", "to", "paper")].edge_index.size(1)),
        ("paper", "to", "author"):
        torch.rand(data[("paper", "to", "author")].edge_index.size(1)),
    } if edge_mask else None

    node_feat_mask = {
        "paper": torch.rand(data["paper"].x.size()),
        "author": torch.rand(data["author"].x.size()),
    } if node_feat_mask else None

    edge_feat_mask = {
        ("paper", "to", "paper"):
        torch.rand(data[("paper", "to", "paper")].edge_attr.size()),
        ("author", "to", "paper"):
        torch.rand(data[("author", "to", "paper")].edge_attr.size()),
        ("paper", "to", "author"):
        torch.rand(data[("paper", "to", "author")].edge_attr.size()),
    } if edge_feat_mask else None

    explanation = HeteroExplanation(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
    )

    explanation.x_dict = data.x_dict
    explanation.edge_index_dict = data.edge_index_dict
    explanation.edge_attr_dict = data.edge_attr_dict

    return explanation


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

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match="but got none"):
        del explanation.node_mask["paper"]
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 8 nodes"):
        explanation = create_random_explanation(data)
        explanation.node_mask["paper"] = torch.randn(5)
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match="but got none"):
        explanation = create_random_explanation(data)
        del explanation.edge_mask[("paper", "to", "paper")]
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match="with 10 edges"):
        explanation = create_random_explanation(data)
        explanation.edge_mask[("paper", "to", "paper")] = torch.randn(5)
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match="but got none"):
        explanation = create_random_explanation(data)
        del explanation.node_feat_mask["paper"]
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match=r"of shape \[8, 16\]"):
        explanation = create_random_explanation(data)
        explanation.node_feat_mask["paper"] = torch.randn(8, 5)
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match="but got none"):
        explanation = create_random_explanation(data)
        del explanation.edge_feat_mask[("paper", "to", "paper")]
        explanation.validate(raise_on_error=True)

    explanation = create_random_explanation(data)
    with pytest.raises(ValueError, match=r"of shape \[10, 16\]"):
        explanation = create_random_explanation(data)
        explanation.edge_feat_mask[("paper", "to",
                                    "paper")] = torch.randn(10, 5)
        explanation.validate(raise_on_error=True)


# TODO: Tests for subgraph and complement


def test_node_mask(data):
    pass


def test_edge_mask(data):
    pass
