import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.data import Data
from torch_geometric.explainability.algo import GNNExplainer
from torch_geometric.nn import GATConv, GCNConv, global_add_pool
from torch_geometric.testing import is_full_test


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, g, **kwargs):
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, g, **kwargs):
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        return x.log_softmax(dim=1)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, g, batch=None, **kwargs):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


return_types = ['log_prob', 'regression']
feat_mask_types = ['individual_feature', 'scalar', 'feature']


@pytest.mark.parametrize('allow_edge_mask', [True, False])
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GNN()])
@pytest.mark.parametrize('task_level', ["graph_level"])
def test_gnn_explainer_explain_graph(model, return_type, allow_edge_mask,
                                     feat_mask_type, task_level):
    explainer = GNNExplainer(log=False, return_type=return_type,
                             allow_edge_mask=allow_edge_mask,
                             feat_mask_type=feat_mask_type)
    explainer.set_objective(lambda x, y, z, i: explainer.loss(x[i], y[i]))

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    edge_attr = torch.randn(edge_index.shape[1], 2)
    y = torch.tensor([3])
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    explanation = explainer.explain(model=model, g=g, target=y,
                                    task_level=task_level)

    if feat_mask_type == 'individual_feature':
        assert explanation.node_features_mask.size() == x.size()
        assert explanation.node_features_mask.min(
        ) >= 0 and explanation.node_features_mask.max() <= 1
    elif feat_mask_type == 'scalar':
        assert explanation.node_mask.size() == (x.size(0), )
        assert explanation.node_mask.min() >= 0 and explanation.node_mask.max(
        ) <= 1
    else:
        assert explanation.node_features_mask.size() == (x.size(1), )
        assert explanation.node_features_mask.min(
        ) >= 0 and explanation.node_features_mask.max() <= 1
    assert explanation.edge_mask.size() == (edge_index.size(1), )
    assert explanation.edge_mask.max() <= 1 and explanation.edge_mask.min(
    ) >= 0

    if is_full_test():
        jit = torch.jit.export(explainer)

        explanation = jit.explain(model=model, g=g, target=y,
                                  task_level=task_level)

        if feat_mask_type == 'individual_feature':
            assert explanation.node_features_mask.size() == x.size()
            assert explanation.node_features_mask.min(
            ) >= 0 and explanation.node_features_mask.max() <= 1
        elif feat_mask_type == 'scalar':
            assert explanation.node_mask.size() == (x.size(0), )
            assert explanation.node_mask.min(
            ) >= 0 and explanation.node_mask.max() <= 1
        else:
            assert explanation.node_features_mask.size() == (x.size(1), )
            assert explanation.node_features_mask.min(
            ) >= 0 and explanation.node_features_mask.max() <= 1
        assert explanation.edge_mask.size() == (edge_index.size(1), )
        assert explanation.edge_mask.max() <= 1 and explanation.edge_mask.min(
        ) >= 0
