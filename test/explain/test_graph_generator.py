import pytest

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
    gen.generate_base_graph()
    return gen


def test_basic_generator(generator):
    assert generator.data.x.size() == (80, 10)
    assert generator.data.edge_index.size(1) >= 400
    assert generator.num_nodes == 80
    assert generator.motif is None


def test_attach_motif(generator):
    generator.num_edges = 2
    generator.num_nodes = 200
    generator.num_motifs = 80
    generator.motif = Motif('house')
    generator.generate_base_graph()
    assert generator.data.x.size() == (600, 10)
    assert generator.data.y.size() == (600, )
    assert generator.data.expl_mask.sum() == 40
    assert generator.data.edge_label.sum() == 960
    assert generator.data.edge_index.size(1) >= 1000


def test_reproducibility():
    pass
