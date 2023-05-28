import torch

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.datasets import InfectionDataset
from torch_geometric.datasets.graph_generator import ERGraph, GraphGenerator


class DummyGraph(GraphGenerator):
    def __call__(self) -> Data:
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8],
        ])
        return Data(num_nodes=10, edge_index=edge_index)


def test_infection_dataset():
    seed_everything(12345)
    graph_generator = DummyGraph()
    dataset = InfectionDataset(graph_generator, num_infected_nodes=2,
                               max_path_length=2)
    assert str(dataset) == ('InfectionDataset(1, '
                            'graph_generator=DummyGraph(), '
                            'num_infected_nodes=2, '
                            'max_path_length=2)')
    assert len(dataset) == 1

    data = dataset[0]
    assert len(data) == 4
    assert data.x.size() == (10, 2)
    assert data.x[:, 0].sum() == 8 and data.x[:, 1].sum() == 2
    assert torch.equal(data.edge_index, graph_generator().edge_index)
    assert data.y.size() == (10, )

    # With `seed=12345`, node 0 and node 7 will be infected:
    assert data.x[0].tolist() == [0, 1]  # First infected node.
    assert data.x[7].tolist() == [0, 1]  # Second infected node.
    assert data.y.tolist() == [0, 1, 2, 3, 3, 2, 1, 0, 1, 2]
    assert data.edge_mask.tolist() == [
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0
    ]


def test_infection_dataset_reproducibility():
    graph_generator = ERGraph(num_nodes=500, edge_prob=0.004)

    seed_everything(12345)
    dataset1 = InfectionDataset(graph_generator, num_infected_nodes=50,
                                max_path_length=5)

    seed_everything(12345)
    dataset2 = InfectionDataset(graph_generator, num_infected_nodes=50,
                                max_path_length=5)

    assert torch.equal(dataset1[0].edge_mask, dataset2[0].edge_mask)
