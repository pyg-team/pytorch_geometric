import pytest
import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif


@pytest.mark.parametrize('graph_generator', [
    BAGraph(num_nodes=80, num_edges=5),
])
@pytest.mark.parametrize('motif_generator', [
    HouseMotif(),
    'house',
])
def test_explainer_dataset_ba_house(graph_generator, motif_generator):
    dataset = ExplainerDataset(graph_generator, motif_generator, num_motifs=2)
    assert str(dataset) == ('ExplainerDataset(1, graph_generator='
                            'BAGraph(num_nodes=80, num_edges=5), '
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
    data1 = ExplainerDataset(BAGraph(num_nodes=80, num_edges=5), HouseMotif(),
                             num_motifs=2)[0]

    seed_everything(12345)
    data2 = ExplainerDataset(BAGraph(num_nodes=80, num_edges=5), HouseMotif(),
                             num_motifs=2)[0]

    assert torch.equal(data1.edge_index, data2.edge_index)
