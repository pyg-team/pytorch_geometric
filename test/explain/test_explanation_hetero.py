import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation

# TODO: Move to a common file and share across all tests.


@pytest.fixture
def data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6, 7]], )
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['paper', 'author'].edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6, 7]], )
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
        "paper": torch.rand(8),
        "author": torch.rand(10)
    } if node_mask else None

    edge_mask = {
        ("paper", "to", "paper"): torch.rand(10),
        ("paper", "to", "author"): torch.rand(10),
    } if edge_mask else None

    node_feat_mask = {
        "paper": torch.rand(8, 16),
        "author": torch.rand(10, 8)
    } if node_feat_mask else None

    edge_feat_mask = {
        ("paper", "to", "paper"): torch.rand(10, 16),
        ("paper", "to", "author"): torch.rand(10, 8),
    } if edge_feat_mask else None

    return HeteroExplanation(
        x_dict=data.x_dict,
        edge_index_dict=data.edge_index_dict,
        edge_attr_dict=data.edge_attr_dict,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
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


def test_node_mask(data):
    node_mask = {
        "paper": torch.ones(8),
        "author": torch.ones(10),
    }

    node_mask["paper"][0] = 0
    node_mask["paper"][1] = 0
    node_mask["author"][4] = 0

    explanation = HeteroExplanation(x_dict=data.x_dict,
                                    edge_index_dict=data.edge_index_dict,
                                    edge_attr_dict=data.edge_attr_dict,
                                    node_mask=node_mask)

    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.x_dict["paper"].size() == (6, 16)
    assert out.x_dict["author"].size() == (9, 8)
    assert out.node_mask["paper"].size() == (6, )
    assert out.node_mask["author"].size() == (9, )
    assert out.edge_index_dict[("paper", "to", "paper")].size() == (2, 4)
    assert out.edge_attr_dict[("paper", "to", "paper")].size() == (4, 16)
    assert out.edge_index_dict[("paper", "to", "author")].size() == (2, 3)
    assert out.edge_attr_dict[("paper", "to", "author")].size() == (3, 8)

    out = explanation.get_complement_subgraph()
    assert out.x_dict["paper"].size() == (2, 16)
    assert out.x_dict["author"].size() == (1, 8)
    assert out.node_mask["paper"].size() == (2, )
    assert out.node_mask["author"].size() == (1, )
    assert out.edge_index_dict[("paper", "to", "paper")].size() == (2, 0)
    assert out.edge_attr_dict[("paper", "to", "paper")].size() == (0, 16)
    assert out.edge_index_dict[("paper", "to", "author")].size() == (2, 2)
    assert out.edge_attr_dict[("paper", "to", "author")].size() == (2, 8)


def test_edge_mask(data):
    edge_mask = {
        ("paper", "to", "paper"): torch.ones(10),
        ("paper", "to", "author"): torch.ones(10),
    }

    edge_mask[("paper", "to", "paper")][0] = 0
    edge_mask[("paper", "to", "paper")][1] = 0
    edge_mask[("paper", "to", "author")][4] = 0

    explanation = HeteroExplanation(x_dict=data.x_dict,
                                    edge_index_dict=data.edge_index_dict,
                                    edge_attr_dict=data.edge_attr_dict,
                                    edge_mask=edge_mask)

    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.x_dict["paper"].size() == (8, 16)
    assert out.x_dict["author"].size() == (10, 8)
    assert out.edge_index_dict[("paper", "to", "paper")].size() == (2, 8)
    assert out.edge_attr_dict[("paper", "to", "paper")].size() == (8, 16)
    assert out.edge_index_dict[("paper", "to", "author")].size() == (2, 9)
    assert out.edge_attr_dict[("paper", "to", "author")].size() == (9, 8)
    assert out.edge_mask[("paper", "to", "paper")].size() == (8, )
    assert out.edge_mask[("paper", "to", "author")].size() == (9, )

    out = explanation.get_complement_subgraph()
    assert out.x_dict["paper"].size() == (8, 16)
    assert out.x_dict["author"].size() == (10, 8)
    assert out.edge_index_dict[("paper", "to", "paper")].size() == (2, 2)
    assert out.edge_attr_dict[("paper", "to", "paper")].size() == (2, 16)
    assert out.edge_index_dict[("paper", "to", "author")].size() == (2, 1)
    assert out.edge_attr_dict[("paper", "to", "author")].size() == (1, 8)
    assert out.edge_mask[("paper", "to", "paper")].size() == (2, )
    assert out.edge_mask[("paper", "to", "author")].size() == (1, )
