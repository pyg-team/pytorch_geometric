import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import GATConv, GCNConv, GNNExplainer, global_add_pool
from torch_geometric.testing import withPackage


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


return_types = ['log_prob', 'regression']
feat_mask_types = ['individual_feature', 'scalar', 'feature']


@withPackage('matplotlib')
@pytest.mark.parametrize('allow_edge_mask', [True, False])
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GCN(), GAT()])
def test_gnn_explainer_explain_node(model, return_type, allow_edge_mask,
                                    feat_mask_type):
    explainer = GNNExplainer(model, log=False, return_type=return_type,
                             allow_edge_mask=allow_edge_mask,
                             feat_mask_type=feat_mask_type)
    assert explainer.__repr__() == 'GNNExplainer()'

    x = torch.randn(8, 3)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

    node_feat_mask, edge_mask = explainer.explain_node(2, x, edge_index)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph(2, edge_index, edge_mask, y=y,
                                            threshold=0.8,
                                            node_alpha=node_feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30, size=(edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph(2, edge_index, edge_mask, y=y,
                                            edge_y=edge_y, threshold=0.8)
    if feat_mask_type == 'individual_feature':
        assert node_feat_mask.size() == x.size()
    elif feat_mask_type == 'scalar':
        assert node_feat_mask.size() == (x.size(0), )
    else:
        assert node_feat_mask.size() == (x.size(1), )
    assert node_feat_mask.min() >= 0 and node_feat_mask.max() <= 1
    assert edge_mask.size() == (edge_index.size(1), )
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1
    if not allow_edge_mask:
        assert edge_mask[:8].tolist() == [1.] * 8
        assert edge_mask[8:].tolist() == [0.] * 6


@withPackage('matplotlib')
@pytest.mark.parametrize('allow_edge_mask', [True, False])
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GNN()])
def test_gnn_explainer_explain_graph(model, return_type, allow_edge_mask,
                                     feat_mask_type):
    explainer = GNNExplainer(model, log=False, return_type=return_type,
                             allow_edge_mask=allow_edge_mask,
                             feat_mask_type=feat_mask_type)

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    edge_attr = torch.randn(edge_index.shape[1], 2)

    node_feat_mask, edge_mask = explainer.explain_graph(
        x, edge_index, edge_attr=edge_attr)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph(-1, edge_index, edge_mask,
                                            y=torch.tensor(2), threshold=0.8,
                                            node_alpha=node_feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30, size=(edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph(-1, edge_index, edge_mask,
                                            edge_y=edge_y, y=torch.tensor(2),
                                            threshold=0.8)
    if feat_mask_type == 'individual_feature':
        assert node_feat_mask.size() == x.size()
    elif feat_mask_type == 'scalar':
        assert node_feat_mask.size() == (x.size(0), )
    else:
        assert node_feat_mask.size() == (x.size(1), )
    assert node_feat_mask.min() >= 0 and node_feat_mask.max() <= 1
    assert edge_mask.size() == (edge_index.size(1), )
    assert edge_mask.max() <= 1 and edge_mask.min() >= 0


@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('model', [GAT()])
def test_gnn_explainer_with_existing_self_loops(model, return_type):
    explainer = GNNExplainer(model, log=False, return_type=return_type)

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

    node_feat_mask, edge_mask = explainer.explain_node(2, x, edge_index)

    assert node_feat_mask.size() == (x.size(1), )
    assert node_feat_mask.min() >= 0 and node_feat_mask.max() <= 1
    assert edge_mask.size() == (edge_index.size(1), )
    assert edge_mask.max() <= 1 and edge_mask.min() >= 0
