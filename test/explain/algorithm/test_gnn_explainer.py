import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
)
from torch_geometric.nn import GATConv, GCNConv, global_add_pool

# --------------------------------------------------------------------
# model for node level tasks
# --------------------------------------------------------------------


class GCN_node_classification_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GCN_node_classification_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.conv2_2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        out_1 = self.conv2(x, edge_index).log_softmax(dim=1)
        out_2 = self.conv2_2(x, edge_index).log_softmax(dim=1)
        return torch.stack([out_1, out_2], dim=0)


class GCN_node_regression_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class GCN_node_regression_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class GAT_node(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


# --------------------------------------------------------------------
# model for edge level tasks
# --------------------------------------------------------------------

# TODO: add edge level models

# --------------------------------------------------------------------
# model for graph level tasks
# --------------------------------------------------------------------


class GCN_graph_classification_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


class GCN_graph_classification_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)
        self.lin2 = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        out_1 = self.lin(x).log_softmax(dim=1)
        out_2 = self.lin2(x).log_softmax(dim=1)
        return torch.stack([out_1, out_2], dim=0)


class GCN_graph_regression_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return self.lin(x)


class GCN_graph_regression_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return self.lin(x)


# --------------------------------------------------------------------
# helper functions and options
# --------------------------------------------------------------------


def check_explanation(
    edge_mask_type,
    node_mask_type,
    x,
    edge_index,
    explanation,
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_features_mask.shape == x.shape
        assert explanation.node_features_mask.min() >= 0
        assert explanation.node_features_mask.max() <= 1
    elif node_mask_type == MaskType.object:
        assert explanation.node_mask.shape == x.shape[0]
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type == MaskType.common_attributes:
        assert explanation.node_features_mask.shape == x.shape
        assert explanation.node_features_mask.min() >= 0
        assert explanation.node_features_mask.max() <= 1
        assert explanation.node_features_mask[
            0] == explanation.node_features_mask[1]

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


# type of masks allowed for GNNExplainer
node_mask_types = ["attributes", "object", "common_attributes"]
edge_mask_types = ["object", None]
return_types_classification = ["log_probs", "raw", "probs"]
return_types_regression = ["raw"]

x = torch.randn(8, 3)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

# --------------------------------------------------------------------
# test for node level tasks
# --------------------------------------------------------------------


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GCN_node_classification_single_output, GCN_node_classification_multi_output
])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_with_meta_explainer_classification_node(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    model = model()

    out = model(x, edge_index).argmax(dim=-1)

    if isinstance(model, GCN_node_classification_multi_output):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=10),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out, index=2,
                            target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize(
    'model',
    [GCN_node_regression_single_output, GCN_node_regression_multi_output])
@pytest.mark.parametrize('return_type', return_types_regression)
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
def test_gnn_explainer_with_meta_explainer_regression_node(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    model = model()
    with torch.no_grad():
        out = model(x, edge_index)

    if isinstance(model, GCN_node_regression_multi_output):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=10),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out, index=2,
                            target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


# --------------------------------------------------------------------
# test for edge level tasks
# --------------------------------------------------------------------

# TODO: add edge level tests

# --------------------------------------------------------------------
# test for graph level tasks
# --------------------------------------------------------------------


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GCN_graph_classification_single_output,
    GCN_graph_classification_multi_output
])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_with_meta_explainer_classification_graph(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="classification",
    )

    model = model()

    out = model(x, edge_index, None, None).argmax(dim=-1)

    if isinstance(model, GCN_graph_classification_multi_output):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=10),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GCN_graph_regression_single_output,
    GCN_graph_regression_multi_output,
])
@pytest.mark.parametrize('return_type', return_types_regression)
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
def test_gnn_explainer_with_meta_explainer_regression_graph(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="regression",
    )

    model = model()
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, None, None)

    if isinstance(model, GCN_graph_regression_multi_output):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=10),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)
