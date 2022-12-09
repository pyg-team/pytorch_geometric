import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.datasets.generators import MotifGenerator
from torch_geometric.testing import withPackage


def test_house_motif():
    generator = MotifGenerator(structure="house")
    house = generator.motif

    assert "Data" in str(house)
    assert len(house) == 3
    assert house.num_features == 0
    assert house.num_edges == 12
    assert house.num_nodes == 5


def test_custom_pyg_motif():
    data = Data(
        num_nodes=3, edge_index=torch.tensor([[0, 1, 2, 1, 2, 0],
                                              [1, 2, 0, 0, 1, 2]]))
    generator = MotifGenerator(structure=data)
    triangle = generator.motif

    assert str(data) == str(triangle)
    assert len(triangle) == len(data)
    assert triangle.edge_index.shape == data.edge_index.shape
    assert torch.equal(triangle.edge_index, data.edge_index)
    assert triangle.num_edges == data.num_edges
    assert triangle.num_nodes == data.num_nodes


@withPackage("networkx")
def test_custom_networkx_motif():
    import networkx as nx

    graph = nx.gnm_random_graph(5, 10, seed=2000)
    generator = MotifGenerator(structure=graph)
    erdos_renyi = generator.motif

    assert "Data" in str(erdos_renyi)
    assert len(erdos_renyi) == 2
    assert erdos_renyi.num_edges == graph.number_of_edges() * 2
    assert erdos_renyi.num_nodes == graph.number_of_nodes()


def test_unknown_motif():
    with pytest.raises(ValueError) as _:
        generator = MotifGenerator(structure="unknown")
        generator.motif
