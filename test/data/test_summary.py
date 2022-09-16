import torch

from torch_geometric.data.summary import Summary
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.testing import withPackage


def test_summary():
    dataset = FakeDataset(num_graphs=10)
    num_nodes = torch.Tensor([data.num_nodes for data in dataset])
    num_edges = torch.Tensor([data.num_edges for data in dataset])

    summary = dataset.get_summary()

    assert summary.name == 'FakeDataset'
    assert summary.num_graphs == 10

    assert summary.num_nodes.mean == num_nodes.mean().item()
    assert summary.num_nodes.std == num_nodes.std().item()
    assert summary.num_nodes.min == num_nodes.min().item()
    assert summary.num_nodes.quantile25 == num_nodes.quantile(0.25).item()
    assert summary.num_nodes.median == num_nodes.median().item()
    assert summary.num_nodes.quantile75 == num_nodes.quantile(0.75).item()
    assert summary.num_nodes.max == num_nodes.max().item()

    assert summary.num_edges.mean == num_edges.mean().item()
    assert summary.num_edges.std == num_edges.std().item()
    assert summary.num_edges.min == num_edges.min().item()
    assert summary.num_edges.quantile25 == num_edges.quantile(0.25).item()
    assert summary.num_edges.median == num_edges.median().item()
    assert summary.num_edges.quantile75 == num_edges.quantile(0.75).item()
    assert summary.num_edges.max == num_edges.max().item()


@withPackage('tabulate')
def test_hetero_summary():
    dataset1 = FakeHeteroDataset(num_graphs=10)
    summary1 = Summary.from_dataset(dataset1)

    dataset2 = [data.to_homogeneous() for data in dataset1]
    summary2 = Summary.from_dataset(dataset2)
    summary2.name = 'FakeHeteroDataset'

    assert summary1 == summary2
    assert str(summary1) == str(summary2)
