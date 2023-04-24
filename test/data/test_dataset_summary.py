import torch
from torch import Tensor

from torch_geometric.data.summary import Stats, Summary
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.testing import withPackage


def check_stats(stats: Stats, expected: Tensor):
    assert stats.mean == float(expected.mean())
    assert stats.std == float(expected.std())
    assert stats.min == float(expected.min())
    assert stats.quantile25 == float(expected.quantile(0.25))
    assert stats.median == float(expected.median())
    assert stats.quantile75 == float(expected.quantile(0.75))
    assert stats.max == float(expected.max())


def test_dataset_summary():
    dataset = FakeDataset(num_graphs=10)
    num_nodes = torch.Tensor([data.num_nodes for data in dataset])
    num_edges = torch.Tensor([data.num_edges for data in dataset])

    summary = dataset.get_summary()

    assert summary.name == 'FakeDataset'
    assert summary.num_graphs == 10

    check_stats(summary.num_nodes, num_nodes)
    check_stats(summary.num_edges, num_edges)


@withPackage('tabulate')
def test_dataset_summary_representation():
    dataset = FakeDataset(num_graphs=10)

    summary1 = Summary.from_dataset(dataset, per_type=False)
    summary2 = Summary.from_dataset(dataset, per_type=True)

    assert str(summary1) == str(summary2)


@withPackage('tabulate')
def test_dataset_summary_hetero():
    dataset1 = FakeHeteroDataset(num_graphs=10)
    summary1 = Summary.from_dataset(dataset1, per_type=False)

    dataset2 = [data.to_homogeneous() for data in dataset1]
    summary2 = Summary.from_dataset(dataset2)
    summary2.name = 'FakeHeteroDataset'

    assert summary1 == summary2
    assert str(summary1) == str(summary2)


@withPackage('tabulate')
def test_dataset_summary_hetero_representation_length():
    dataset = FakeHeteroDataset(num_graphs=10)
    summary = Summary.from_dataset(dataset)
    num_lines = len(str(summary).splitlines())

    stats_len = len(Stats.__dataclass_fields__)
    len_header_and_border = 5
    num_tables = 3  # general, stats per node type, stats per edge type

    assert num_lines == num_tables * (stats_len + len_header_and_border)


def test_dataset_summary_hetero_per_type_check():
    dataset = FakeHeteroDataset(num_graphs=10)
    exp_num_nodes = torch.Tensor([data.num_nodes for data in dataset])
    exp_num_edges = torch.Tensor([data.num_edges for data in dataset])

    summary = dataset.get_summary()

    assert summary.name == 'FakeHeteroDataset'
    assert summary.num_graphs == 10

    check_stats(summary.num_nodes, exp_num_nodes)
    check_stats(summary.num_edges, exp_num_edges)

    num_nodes_per_type = {}
    for node_type in dataset.node_types:
        num_nodes_per_type[node_type] = torch.Tensor(
            [data[node_type].num_nodes for data in dataset])

    assert len(summary.num_nodes_per_type) == len(dataset.node_types)
    for node_type, stats in summary.num_nodes_per_type.items():
        check_stats(stats, num_nodes_per_type[node_type])

    num_edges_per_type = {}
    for edge_type in dataset.edge_types:
        num_edges_per_type[edge_type] = torch.Tensor(
            [data[edge_type].num_edges for data in dataset])

    assert len(summary.num_edges_per_type) == len(dataset.edge_types)
    for edge_type, stats in summary.num_edges_per_type.items():
        check_stats(stats, num_edges_per_type[edge_type])
