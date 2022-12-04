import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.datasets.generators import ERGraph, Motif

@pytest.mark.parametrize('num_nodes', [10, 50, 100])
@pytest.mark.parametrize('edge_prob', [0.1, 0.2, 0.9])
@pytest.mark.parametrize('num_motifs', [10, 50, 100])
def test(num_nodes, edge_prob, num_motifs):
    # check if initializes correctly
    motif = Motif(structure="house")
    er_graph_gen = ERGraph(
        motif=motif, num_nodes=num_nodes, edge_prob=edge_prob, num_motifs=num_motifs
    )

    # check if it has `generate_base_graph()` method
    assert hasattr(er_graph_gen, 'generate_base_graph')

    # check if graph generates base_graph
    data = er_graph_gen.generate_base_graph()

    # check if data has correct edge_index
    assert torch.eq(data.edge_index, erdos_renyi_graph(num_nodes, edge_prob))
    # check if data is of torch_geometric.data.Data type
    assert isinstance(data, Data)
