import pytest
import torch
from torch import Tensor
from torch.nn import Linear

from torch_geometric.explain import Explainer, Explanation, GNNExplainer
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
)
from torch_geometric.nn import GATConv, GCNConv, global_add_pool

# Models for node level tasks #################################################


class GCN_node_classification_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GCN_node_classification_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.conv2_2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        out1 = self.conv2(x, edge_index).log_softmax(dim=1)
        out2 = self.conv2_2(x, edge_index).log_softmax(dim=1)
        return torch.stack([out1, out2], dim=0)


class GCN_node_regression_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GCN_node_regression_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GAT_node(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


# Models for edge level tasks ################################################


class GCN_edge_classification_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.linear1 = Linear(14, 4)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        edge_x = torch.cat([x[src], x[dst]], dim=1)
        return self.linear1(edge_x).log_softmax(dim=1)


class GCN_edge_classification_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.linear1 = Linear(14, 4)
        self.linear2 = Linear(14, 4)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        edge_x = torch.cat([x[src], x[dst]], dim=1)
        out_1 = self.linear1(edge_x).log_softmax(dim=1)
        out_2 = self.linear2(edge_x).log_softmax(dim=1)
        return torch.stack([out_1, out_2], dim=0)


class GCN_edge_regression_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.linear1 = Linear(14, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        edge_x = torch.cat([x[src], x[dst]], dim=1)
        return self.linear1(edge_x)


class GCN_edge_regression_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.linear1 = Linear(14, 4)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        edge_x = torch.cat([x[src], x[dst]], dim=1)
        return self.linear1(edge_x)


# Models for graph level tasks ################################################


class GCN_graph_classification_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # `edge_attr` is used as a dummy to check if the explainer passes
        # arguments in the right order.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
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
        # `edge_attr` is used as a dummy to check if the explainer passes
        # arguments in the right order.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
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
        # `edge_attr` is used as a dummy to check if the explainer passes
        # arguments in the right order.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.lin(x)


class GCN_graph_regression_multi_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # `edge_attr` is used as a dummy to check if the explainer passes
        # arguments in the right order.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.lin(x)


def check_explanation(
    edge_mask_type: MaskType,
    node_mask_type: MaskType,
    x: Tensor,
    edge_index: Tensor,
    explanation: Explanation,
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_feat_mask.size() == x.size()
        assert explanation.node_feat_mask.min() >= 0
        assert explanation.node_feat_mask.max() <= 1
    elif node_mask_type == MaskType.object:
        assert explanation.node_mask.size() == (x.size(0), )
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type == MaskType.common_attributes:
        assert explanation.node_feat_mask.size() == x.size()
        assert explanation.node_feat_mask.min() >= 0
        assert explanation.node_feat_mask.max() <= 1
        assert torch.allclose(explanation.node_feat_mask[0],
                              explanation.node_feat_mask[1])

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.size() == (edge_index.size(1), )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


node_mask_types = [
    MaskType.object,
    MaskType.common_attributes,
    MaskType.attributes,
]
edge_mask_types = [MaskType.object, None]
return_types_classification = ['log_probs', 'raw', 'probs']
return_types_regression = ['raw']

x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
edge_attr = torch.randn(edge_index.size(1), 2)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('Model', [
    GCN_node_classification_single_output, GCN_node_classification_multi_output
])
@pytest.mark.parametrize("explanation_type", ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_classification)
@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_gnn_explainer_classification_node(
    edge_mask_type,
    node_mask_type,
    Model,
    explanation_type,
    return_type,
    index,
):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index).argmax(dim=-1)

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ), model_config=ModelConfig(
            task_level='node',
            return_type=return_type,
            mode='classification',
        ))

    if isinstance(model, GCN_node_classification_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(x, edge_index, target=out, index=index,
                            target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('Model', [
    GCN_node_regression_single_output,
    GCN_node_regression_multi_output,
])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_regression)
@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_gnn_explainer_regression_node(edge_mask_type, node_mask_type, Model,
                                       explanation_type, return_type, index):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ),
        model_config=ModelConfig(
            task_level='node',
            return_type=return_type,
            mode='regression',
        ),
    )

    if isinstance(model, GCN_node_regression_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            index=index, target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('Model', [
    GCN_edge_classification_single_output, GCN_edge_classification_multi_output
])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_classification_edge(
    edge_mask_type,
    node_mask_type,
    Model,
    explanation_type,
    return_type,
):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr).argmax(dim=-1)

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ), model_config=ModelConfig(
            task_level='edge',
            return_type=return_type,
            mode='classification',
        ))

    if isinstance(model, GCN_edge_classification_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(x=x, edge_index=edge_index, target=out, index=0,
                            target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('Model', [
    GCN_edge_regression_single_output,
    GCN_edge_regression_multi_output,
])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_regression_edge(
    edge_mask_type,
    node_mask_type,
    Model,
    explanation_type,
    return_type,
):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr)

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ), model_config=ModelConfig(
            task_level='edge',
            return_type=return_type,
            mode='regression',
        ))

    if isinstance(model, GCN_edge_regression_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(x=x, edge_index=edge_index, target=out, index=0,
                            target_index=target_index)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('Model', [
    GCN_graph_classification_single_output,
    GCN_graph_classification_multi_output
])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_with_meta_explainer_classification_graph(
    edge_mask_type,
    node_mask_type,
    Model,
    explanation_type,
    return_type,
):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr).argmax(dim=-1)

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ), model_config=ModelConfig(
            task_level='graph',
            return_type=return_type,
            mode='classification',
        ))

    if isinstance(model, GCN_graph_classification_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        target=out,
        target_index=target_index,
        edge_attr=None,
        batch=None,
    )

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize(
    'Model',
    [GCN_graph_regression_single_output, GCN_graph_regression_multi_output])
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_with_meta_explainer_regression_graph(
    edge_mask_type,
    node_mask_type,
    Model,
    explanation_type,
    return_type,
):
    model = Model()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr)

    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=2),
        explainer_config=ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        ), model_config=ModelConfig(
            task_level='graph',
            return_type=return_type,
            mode='regression',
        ))

    if isinstance(model, GCN_graph_regression_multi_output):
        target_index = 0
    else:
        target_index = None

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        target=out,
        target_index=target_index,
        edge_attr=None,
        batch=None,
    )

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)
