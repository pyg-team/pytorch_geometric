import pytest
import torch

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.generators import Motif
from torch_geometric.datasets.generators.graph_generator import GraphGenerator
from torch_geometric.utils import barabasi_albert_graph


class BAGraph(GraphGenerator):
    def __init__(
        self,
        num_nodes: int = 80,
        num_edges: int = 5,
        num_motifs: int = 20,
        motif=None,
    ):
        self.num_edges = num_edges
        self.num_motifs = num_motifs
        super().__init__(num_nodes, motif)

    def generate_base_graph(self):
        self._edge_index = barabasi_albert_graph(self.num_nodes,
                                                 self.num_edges)
        self.attach_motif()
        self.generate_feature()


@pytest.fixture
def generator():
    gen = BAGraph(num_nodes=80)
    return gen


def test_basic_generator(generator):
    dataset = ExplainerDataset(generator)
    data = dataset[0]
    assert data.x.size() == (80, 10)
    assert data.edge_index.size(1) >= 400
    assert generator.num_nodes == 80
    assert generator.motif is None


def test_attach_motif(generator):
    generator.num_edges = 2
    generator.num_nodes = 200
    generator.num_motifs = 80
    generator.motif = Motif('house')
    dataset = ExplainerDataset(generator)
    data = dataset[0]
    assert data.x.size() == (600, 10)
    assert data.y.size() == (600, )
    assert data.expl_mask.sum() == 40
    assert data.edge_label.sum() == 960
    assert data.edge_index.size(1) >= 1000


def test_reproducibility(generator):
    generator.num_nodes = 100
    generator.num_motifs = 5
    generator.motif = Motif('house')
    generator.seed = 123
    data1 = ExplainerDataset(generator)[0]
    generator.seed = 44
    generator.num_nodes = 100
    data2 = ExplainerDataset(generator)[0]
    generator.seed = 123
    generator.num_nodes = 100
    data3 = ExplainerDataset(generator)[0]
    assert torch.all(data1.edge_index == data3.edge_index)
    assert not torch.all(data1.edge_index == data2.edge_index)
