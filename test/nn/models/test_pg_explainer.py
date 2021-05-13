import pytest
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import (GCNConv, GATConv, PGExplainer,
                                MessagePassing, global_add_pool)


def assert_edge_mask(model):
    for layer in model.modules():
        if isinstance(layer, MessagePassing):
            assert ~layer.__explain__
            assert layer.__edge_mask__ is None


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = Sequential(Linear(16, 16), ReLU(), Dropout(0.2),
                              Linear(16, 7))

    def forward(self, x, edge_index, batch, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        x = global_add_pool(x, batch)
        x = self.fc1(x)
        return x.log_softmax(dim=1)


@pytest.mark.parametrize('model', [Net()])
def test_graph(model):
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    z = model(x, edge_index, batch, get_embedding=True)

    explainer = PGExplainer(model, out_channels=z.shape[1], task='graph')
    explainer.train_explainer(x, z, edge_index, batch=batch)
    assert_edge_mask(model)

    mask = explainer.explain(x, z, edge_index)
    assert mask.shape[0] == edge_index.shape[1]
    assert mask.max() <= 1 and mask.min() >= 0


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        return x.log_softmax(dim=1)


@pytest.mark.parametrize('model', [GAT(), GCN()])
def test_node(model):

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    z = model(x, edge_index, get_embedding=True)

    explainer = PGExplainer(model, out_channels=z.shape[1], task='node')
    explainer.train_explainer(x, z, edge_index, node_idxs=torch.arange(0, 3))
    assert_edge_mask(model)

    mask = explainer.explain(x, z, edge_index, node_id=1)
    assert mask.shape[0] == edge_index.shape[1]
    assert mask.max() <= 1 and mask.min() >= 0
