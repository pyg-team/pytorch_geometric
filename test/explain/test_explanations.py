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


@pytest.fixture
def small_data():
    return Data(
        x=torch.randn(4, 3), edge_index=torch.tensor([[0, 0, 0, 1, 1, 2],
                                                      [1, 2, 3, 2, 3, 3]]),
        edge_attr=torch.randn(6, 3))


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


def test_node_mask(small_data):
    node_mask = torch.tensor([True, False, True, False])
    node_feat_mask = torch.zeros(4, 3)
    node_feat_mask[:, 1] = 1
    explanation = Explanation(node_mask=node_mask,
                              node_feat_mask=node_feat_mask.type(torch.bool),
                              x=small_data.x, edge_index=small_data.edge_index,
                              edge_attr=small_data.edge_attr)
    explanation.validate(raise_on_error=True)

    # test explanation graph
    # removing node 0 and 2 removes 5 edges
    # the node_feat_mask only leaves feature 1 nonzero
    explanation_graph = explanation.get_explanation_subgraph()
    assert explanation_graph.x.shape == (2, 3)
    assert explanation_graph.edge_index.shape == (2, 1)
    assert explanation_graph.edge_attr.shape == (1, 3)
    assert torch.allclose(explanation_graph.x[:, 0], torch.zeros(2))
    assert torch.allclose(explanation_graph.x[:, 2], torch.zeros(2))

    # test complement graph
    # removing node 1 and 3 removes 5 edges
    # the node_feat_mask sets feature 1 to zero
    complement_graph = explanation.get_complement_subgraph()
    assert complement_graph.x.shape == (2, 3)
    assert complement_graph.edge_index.shape == (2, 1)
    assert complement_graph.edge_attr.shape == (1, 3)
    assert torch.allclose(complement_graph.x[:, 1], torch.zeros(2))


def test_edge_mask(small_data):
    edge_mask = torch.tensor([True, False, True, False, False, True])
    edge_feat_mask = torch.zeros(6, 3)
    edge_feat_mask[:, 1] = 1
    explanation = Explanation(edge_mask=edge_mask,
                              edge_feat_mask=edge_feat_mask.type(torch.bool),
                              x=small_data.x, edge_index=small_data.edge_index,
                              edge_attr=small_data.edge_attr)
    explanation.validate(raise_on_error=True)

    # test explanation graph
    # removing edges does not change the nodes
    # the edge_feat_mask only leaves feature 1 nonzero
    explanation_graph = explanation.get_explanation_subgraph()
    assert explanation_graph.x.shape == (4, 3)
    assert explanation_graph.edge_index.shape == (2, 3)
    assert explanation_graph.edge_attr.shape == (3, 3)
    assert torch.allclose(explanation_graph.edge_attr[:, 0], torch.zeros(3))
    assert torch.allclose(explanation_graph.edge_attr[:, 2], torch.zeros(3))

    # test complement graph
    # removing edges does not change the nodes
    # the edge_feat_mask sets feature 1 to zero
    complement_graph = explanation.get_complement_subgraph()
    assert complement_graph.x.shape == (4, 3)
    assert complement_graph.edge_index.shape == (2, 3)
    assert complement_graph.edge_attr.shape == (3, 3)
    assert torch.allclose(complement_graph.edge_attr[:, 1], torch.zeros(3))


def test_full_explanation_mask(small_data):
    node_mask = torch.tensor([True, False, False, False])
    node_feat_mask = torch.zeros(4, 3)
    node_feat_mask[:, 1] = 1

    edge_mask = torch.tensor([True, False, True, False, False, True])
    edge_feat_mask = torch.zeros(6, 3)
    edge_feat_mask[:, 1] = 1

    explanation = Explanation(node_mask=node_mask, edge_mask=edge_mask,
                              node_feat_mask=node_feat_mask.type(torch.bool),
                              edge_feat_mask=edge_feat_mask.type(torch.bool),
                              x=small_data.x, edge_index=small_data.edge_index,
                              edge_attr=small_data.edge_attr)

    # explanation
    explanation_graph = explanation.get_explanation_subgraph()
    assert explanation_graph.x.shape == (1, 3)
    assert explanation_graph.edge_index.shape == (2, 0)
    assert explanation_graph.edge_attr.shape == (0, 3)
    assert torch.allclose(explanation_graph.x[:, 0], torch.zeros(1))
    assert torch.allclose(explanation_graph.x[:, 2], torch.zeros(1))

    # complement
    complement_graph = explanation.get_complement_subgraph()
    assert complement_graph.x.shape == (3, 3)
    assert complement_graph.edge_index.shape == (2, 2)
    assert complement_graph.edge_attr.shape == (2, 3)
    assert torch.allclose(complement_graph.edge_attr[:, 0], torch.zeros(2))
    assert torch.allclose(complement_graph.edge_attr[:, 2], torch.zeros(2))
    assert torch.allclose(complement_graph.x[:, 0], torch.zeros(3))
    assert torch.allclose(complement_graph.x[:, 2], torch.zeros(3))


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
