import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain.utils import Interface
from torch_geometric.nn import GAT, GCN

x = torch.randn(8, 3)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
edge_attr = torch.randn(edge_index.shape[1], 2)
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
GCN_model = GCN(3, 16, 2, 7)
GAT_model = GAT(3, 16, 2, 7, heads=2, concat=False)


class GCNCcompatible(GCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface = Interface(graph_to_inputs=lambda x: {
            "x": x.x,
            "edge_index": x.edge_index
        })

    def forward(self, g, **kwargs):
        return super().forward(**self.interface.graph_to_inputs(g))


class GATCcompatible(GAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface = Interface(graph_to_inputs=lambda x: {
            "x": x.x,
            "edge_index": x.edge_index
        })

    def forward(self, g, **kwargs):
        return super().forward(**self.interface.graph_to_inputs(g))


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
@pytest.mark.parametrize("model", [GCN_model, GAT_model])
@pytest.mark.parametrize("kwargs", [{}, {
    "output_idx": None
}, {
    "output_idx": 1
}])
def test_model_with_interface(x, edge_index, edge_attr, graph, model, kwargs):
    interface = Interface()

    out = model(x=x, edge_index=edge_index)
    if kwargs:
        with pytest.raises(TypeError):
            out_tilde = model(**interface.graph_to_inputs(graph, **kwargs))
    else:
        out_tilde = model(**interface.graph_to_inputs(graph, **kwargs))
        assert torch.all(out == out_tilde)


@pytest.mark.parametrize("model", [(GCN_model, GCNCcompatible(3, 16, 2, 7)),
                                   (GAT_model, GATCcompatible(3, 16, 2, 7))])
@pytest.mark.parametrize("x", [x])
@pytest.mark.parametrize("edge_index", [edge_index])
@pytest.mark.parametrize("graph", [graph])
def def_test_model_interface(x, edge_index, graph, model):
    model, model_compatible = model
    model_compatible.load_state_dict(model.state_dict())
    out = model(x=x, edge_index=edge_index)
    out_tilde = model_compatible(graph)
    assert torch.all(out == out_tilde)
