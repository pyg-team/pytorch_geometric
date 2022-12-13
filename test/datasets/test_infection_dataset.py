import pytest

from torch_geometric.datasets import InfectionDataset
from torch_geometric.datasets.graph_generator import ERGraph


@pytest.mark.parametrize('graph_generator', [
    ERGraph(num_nodes=500, edge_prob=0.004),
])
def test_infection_dataset_er(graph_generator):
    dataset = InfectionDataset(graph_generator, seeds=[12345])
    assert str(dataset) == ('InfectionDataset(1, graph_generator='
                            'ERGraph(num_nodes=500, edge_prob=0.004)')
    assert len(dataset) == 1
    data1 = dataset[0]

    assert len(data1) == 5
    assert data1.num_nodes == 500
    assert data1.edge_index.min() == 0
    assert data1.edge_index.max() == data1.num_nodes - 1
    assert data1.y.min() == 0 and data1.y.max() == 5
    assert data1.node_mask.size() == (data1.num_nodes, )
    assert data1.node_mask.min() == 0 and data1.node_mask.max() == 1
    assert data1.node_mask.sum() == 242
    assert len(data1.explain_path_edge_index) == 242

    data2 = InfectionDataset(graph_generator, seeds=[12345])[0]
    assert data1.node_mask.sum() == data2.node_mask.sum()
    assert len(data1.explain_path_edge_index) == len(
        data2.explain_path_edge_index)
