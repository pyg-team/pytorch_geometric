import pytest
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Entities, Planetoid, TUDataset
from torch_geometric.nn import GraphMaskExplainer, global_add_pool
from torch_geometric.nn.conv import FastRGCNConv, GATConv, GCNConv
from torch_geometric.testing import withPackage


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 16, normalize=False)
        self.conv2 = GCNConv(16, 7, normalize=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(1433, 8, heads=8)
        self.conv2 = GATConv(64, 7, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastRGCNConv(4, 8, 90, num_blocks=2, aggr='mean')
        self.conv2 = FastRGCNConv(8, 12, 90, num_blocks=4, aggr='add')
        self.conv3 = FastRGCNConv(12, 4, 90, num_blocks=2, aggr='mean')

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


class GCNGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 7, add_self_loops=True, normalize=False)
        self.conv2 = GCNConv(7, 9, add_self_loops=False, normalize=True)
        self.conv3 = GCNConv(9, 6, add_self_loops=True, normalize=False)

    def forward(self, x, edge_index, batch, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=1)


return_types = ['log_prob', 'regression']
feat_mask_types = ['individual_feature', 'scalar', 'feature']


@withPackage('matplotlib')
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GCN()])
def test_graphmask_explainer_gcn_explain_node(model, return_type,
                                              feat_mask_type):
    explainer = GraphMaskExplainer(2, model_to_explain=model, log=False,
                                   return_type=return_type,
                                   feat_mask_type=feat_mask_type,
                                   allow_multiple_explanations=True)
    assert explainer.__repr__() == 'GraphMaskExplainer()'

    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
    data = dataset[0]
    feat_mask = explainer.train_node_explainer([6, 12], data.x,
                                               data.edge_index)
    edge_mask = explainer.explain_node([6, 12], data.x, data.edge_index)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph([6, 12], data.edge_index,
                                            edge_mask, y=data.y,
                                            node_alpha=feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30,
                               size=(data.edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph([6, 12], data.edge_index,
                                            edge_mask, y=data.y, edge_y=edge_y)
    if feat_mask_type == 'individual_feature':
        assert feat_mask.size() == data.x.size()
    elif feat_mask_type == 'scalar':
        assert feat_mask.size() == (data.x.size(0), )
    else:
        assert feat_mask.size() == (data.x.size(1), )
    assert feat_mask.min() >= 0 and feat_mask.max() <= 1
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1


@withPackage('matplotlib')
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GAT()])
def test_graphmask_explainer_gat_explain_node(model, return_type,
                                              feat_mask_type):
    explainer = GraphMaskExplainer(2, model_to_explain=model, log=False,
                                   return_type=return_type,
                                   feat_mask_type=feat_mask_type,
                                   layer_type='GAT',
                                   allow_multiple_explanations=True)
    assert explainer.__repr__() == 'GraphMaskExplainer()'

    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
    data = dataset[0]
    feat_mask = explainer.train_node_explainer([6, 12], data.x,
                                               data.edge_index)
    edge_mask = explainer.explain_node([6, 12], data.x, data.edge_index)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph([6, 12], data.edge_index,
                                            edge_mask, y=data.y,
                                            node_alpha=feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30,
                               size=(data.edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph([6, 12], data.edge_index,
                                            edge_mask, y=data.y, edge_y=edge_y)
    if feat_mask_type == 'individual_feature':
        assert feat_mask.size() == data.x.size()
    elif feat_mask_type == 'scalar':
        assert feat_mask.size() == (data.x.size(0), )
    else:
        assert feat_mask.size() == (data.x.size(1), )
    assert feat_mask.min() >= 0 and feat_mask.max() <= 1
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1


@withPackage('matplotlib')
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [RGCN()])
def test_graphmask_explainer_rgcn_explain_node(model, return_type,
                                               feat_mask_type):
    explainer = GraphMaskExplainer(3, model_to_explain=model, log=False,
                                   return_type=return_type, epochs=5,
                                   feat_mask_type=feat_mask_type,
                                   layer_type='FastRGCN',
                                   allow_multiple_explanations=True)
    assert explainer.__repr__() == 'GraphMaskExplainer()'

    dataset = Entities(root='/tmp/AIFB', name='AIFB')
    edge_index, edge_type = dataset[0].edge_index, dataset[0].edge_type
    x = torch.randn(8285, 4)
    feat_mask = explainer.train_node_explainer([1, 5, 12], x, edge_index,
                                               edge_type=edge_type)
    edge_mask = explainer.explain_node([1, 5, 12], x, edge_index)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph([1, 5, 12], edge_index, edge_mask,
                                            node_alpha=feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30, size=(edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph([1, 5, 12], edge_index, edge_mask,
                                            edge_y=edge_y)
    if feat_mask_type == 'individual_feature':
        assert feat_mask.size() == x.size()
    elif feat_mask_type == 'scalar':
        assert feat_mask.size() == (x.size(0), )
    else:
        assert feat_mask.size() == (x.size(1), )
    assert feat_mask.min() >= 0 and feat_mask.max() <= 1
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1


@withPackage('matplotlib')
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('feat_mask_type', feat_mask_types)
@pytest.mark.parametrize('model', [GCNGraph()])
def test_graphmask_explainer_explain_graph(model, return_type, feat_mask_type):
    explainer = GraphMaskExplainer(3, model, log=False,
                                   return_type=return_type, epochs=5,
                                   feat_mask_type=feat_mask_type)

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    edge_weight = torch.rand(dataset[4].edge_index.size(1))
    feat_mask = explainer.train_graph_explainer(4, dataset[4].x,
                                                dataset[4].edge_index,
                                                edge_weight=edge_weight)
    edge_mask = explainer.explain_graph(4)
    if feat_mask_type == 'scalar':
        _, _ = explainer.visualize_subgraph(None, dataset[4].edge_index,
                                            edge_mask, y=dataset[4].y,
                                            task='graph', node_alpha=feat_mask)
    else:
        edge_y = torch.randint(low=0, high=30,
                               size=(dataset[4].edge_index.size(1), ))
        _, _ = explainer.visualize_subgraph(None, dataset[4].edge_index,
                                            edge_mask, edge_y=edge_y,
                                            y=dataset[4].y, task='graph')
    if feat_mask_type == 'individual_feature':
        assert feat_mask.size() == dataset[4].x.size()
    elif feat_mask_type == 'scalar':
        assert feat_mask.size() == (dataset[4].x.size(0), )
    else:
        assert feat_mask.size() == (dataset[4].x.size(1), )
    assert feat_mask.min() >= 0 and feat_mask.max() <= 1
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1
