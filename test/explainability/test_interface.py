import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explainability.utils import Interface
from torch_geometric.nn import GAT, GCN

x = torch.randn(8, 3)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
edge_attr = torch.randn(edge_index.shape[1], 2)
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
GCN = GCN(3, 16, 2, 7)
GAT = GAT(3, 16, 2, 7, heads=2, concat=False)


@pytest.mark.parametrize("x", [x])
@pytest.mark.parametrize("edge_index", [edge_index])
@pytest.mark.parametrize("edge_attr", [edge_attr])
@pytest.mark.parametrize("graph", [graph])
def test_interface_simple_graph(x, edge_index, edge_attr, graph):
    interface = Interface()
    dict_inputs = interface.graph_to_inputs(graph)
    for key in graph.keys:
        assert torch.allclose(dict_inputs[key], graph[key])

    graph_tilde = interface.inputs_to_graph(x, edge_index, edge_attr)
    assert torch.all(graph_tilde.x == graph.x)
    assert torch.all(graph_tilde.edge_index == graph.edge_index)
    assert torch.all(graph_tilde.edge_attr == graph.edge_attr)


@pytest.mark.parametrize("x", [x])
@pytest.mark.parametrize("edge_index", [edge_index])
@pytest.mark.parametrize("edge_attr", [edge_attr])
@pytest.mark.parametrize("graph", [graph])
@pytest.mark.parametrize("model", [GCN, GAT])
def test_model_with_interface(x, edge_index, edge_attr, graph, model):
    interface = Interface()
    out = model(x=x, edge_index=edge_index)
    out_tilde = model(**interface.graph_to_inputs(graph))
    assert torch.all(out == out_tilde), out - out_tilde
