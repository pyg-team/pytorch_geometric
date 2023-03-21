from copy import deepcopy

import pytest
import torch
from torch.nn import Module

from torch_geometric.contrib.nn import GLTModel, GLTMask, GLTSearch
from torch_geometric.nn import GCN, GIN, GAT
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],
                           [3, 4],
                           [4, 3],
                           [1, 3],
                           [3, 1],
                           [0, 5],
                           [5, 0]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [-1], [0], [1]], dtype=torch.float)

train_mask = torch.tensor([[True], [True], [False], [False], [False], [False]])
val_mask = torch.tensor([[False], [False], [True], [True], [False], [False]])
test_mask = torch.tensor([[False], [False], [False], [False], [True], [True]])


def generate_edge_data(dataset):
    """sample  negative edges and labels"""
    negative_edges = negative_sampling(dataset.edge_index)
    edge_labels = [0] * negative_edges.shape[-1] + [1] * dataset.edge_index.shape[-1]
    dataset.edges = torch.cat([dataset.edge_index, negative_edges], dim=-1)
    dataset.edge_labels = torch.tensor(edge_labels)


test_data_node_classification = Data(x=x, edge_index=edge_index.t().contiguous())
test_data_link_prediction = deepcopy(test_data_node_classification)
generate_edge_data(test_data_node_classification)


class LinkPredictor(Module):
    """helper model to get dot product interaction on edges"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_weight, edges):
        x = self.model(x, edge_index, edge_weight=edge_weight)
        edge_feat_i = x[edges[0]]
        edge_feat_j = x[edges[1]]
        return (edge_feat_i * edge_feat_j).sum(dim=-1)


def build_test_model(architecture, link_prediction=False):
    input_dim = 1
    hidden_dim = 5
    if link_prediction:
        model = architecture(input_dim, hidden_dim, hidden_dim)
        model = LinkPredictor(model)
    else:
        model = architecture(input_dim, hidden_dim, 1)
    return model


@pytest.mark.parametrize('architecture', [GCN, GIN, GAT])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_GLTModel_forward(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
        input_shape = graph.x.shape
    else:
        graph = test_data_node_classification
        input_shape = graph.edges.shape

    model = build_test_model(architecture, link_prediction)
    model = GLTModel(model, graph)
    output = model()

    assert output.shape[0] == input_shape[0]


