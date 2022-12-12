import pytest
import torch

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.utils import barabasi_albert_graph


# TODO Remove after the BAGraph is merged.
class BAGraph(GraphGenerator):
    def __call__(self):
        edge_index = barabasi_albert_graph(num_nodes=80, num_edges=5)
        return Data(num_nodes=80, edge_index=edge_index)


@pytest.mark.parametrize('graph_generator', [BAGraph()])
@pytest.mark.parametrize('motif_generator', [HouseMotif(), 'house'])
def test_explainer_dataset_ba_house(graph_generator, motif_generator):
    dataset = ExplainerDataset(graph_generator, motif_generator, num_motifs=2)
    assert str(dataset) == ('ExplainerDataset(1, graph_generator=BAGraph(), '
                            'motif_generator=HouseMotif(), num_motifs=2)')
    assert len(dataset) == 1

    data = dataset[0]
    assert len(data) == 4
    assert data.num_nodes == 80 + (2 * 5)
    assert data.edge_index.min() == 0
    assert data.edge_index.max() == data.num_nodes - 1
    assert data.y.min() == 0 and data.y.max() == 3
    assert data.node_mask.size() == (data.num_nodes, )
    assert data.edge_mask.size() == (data.num_edges, )
    assert data.node_mask.min() == 0 and data.node_mask.max() == 1
    assert data.node_mask.sum() == 2 * 5
    assert data.edge_mask.min() == 0 and data.edge_mask.max() == 1
    assert data.edge_mask.sum() == 2 * 12


def test_explainer_dataset_reproducibility():
    seed_everything(12345)
    dataset = ExplainerDataset(BAGraph(), HouseMotif(), num_motifs=2)
    data1 = dataset[0]

    seed_everything(12345)
    dataset = ExplainerDataset(BAGraph(), HouseMotif(), num_motifs=2)
    data2 = dataset[0]

    assert torch.equal(data1.edge_index, data2.edge_index)
