import torch

from torch_geometric.data.summary import Stats, Summary
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.testing import withPackage


def check_summary_item(summary_item, exp_item):
    assert summary_item.mean == exp_item.mean().item()
    assert summary_item.std == exp_item.std().item()
    assert summary_item.min == exp_item.min().item()
    assert summary_item.quantile25 == exp_item.quantile(0.25).item()
    assert summary_item.median == exp_item.median().item()
    assert summary_item.quantile75 == exp_item.quantile(0.75).item()
    assert summary_item.max == exp_item.max().item()


def test_dataset_summary():
    dataset = FakeDataset(num_graphs=10)
    num_nodes = torch.Tensor([data.num_nodes for data in dataset])
    num_edges = torch.Tensor([data.num_edges for data in dataset])

    summary = dataset.get_summary()

    assert summary.name == 'FakeDataset'
    assert summary.num_graphs == 10

    check_summary_item(summary.num_nodes, num_nodes)
    check_summary_item(summary.num_edges, num_edges)


@withPackage('tabulate')
def test_dataset_summary_representation():
    dataset = FakeDataset(num_graphs=10)

    summary1 = Summary.from_dataset(dataset, per_type_breakdown=False)
    summary2 = Summary.from_dataset(dataset, progress_bar=True,
                                    per_type_breakdown=True)

    assert str(summary1) == str(summary2)


@withPackage('tabulate')
def test_dataset_summary_hetero():
    dataset1 = FakeHeteroDataset(num_graphs=10)
    summary1 = Summary.from_dataset(dataset1, per_type_breakdown=False)

    dataset2 = [data.to_homogeneous() for data in dataset1]
    summary2 = Summary.from_dataset(dataset2)
    summary2.name = 'FakeHeteroDataset'

    assert summary1 == summary2
    assert str(summary1) == str(summary2)


@withPackage('tabulate')
def test_dataset_summary_hetero_representation_len():
    dataset = FakeHeteroDataset(num_graphs=10)
    summary = Summary.from_dataset(dataset, progress_bar=True)
    summ_num_lines = len(str(summary).splitlines())

    stats_len = len(Stats.__dataclass_fields__)
    header_and_border_len = 5
    summ_tables_count = 3  # general, node per type, edge per type
    exp_length = summ_tables_count * (stats_len + header_and_border_len)

    assert summ_num_lines == exp_length


def test_dataset_summary_hetero_per_type_check():
    dataset = FakeHeteroDataset(num_graphs=10)
    exp_num_nodes = torch.Tensor([data.num_nodes for data in dataset])
    exp_num_edges = torch.Tensor([data.num_edges for data in dataset])

    summary = dataset.get_summary()

    assert summary.name == 'FakeHeteroDataset'
    assert summary.num_graphs == 10

    check_summary_item(summary.num_nodes, exp_num_nodes)
    check_summary_item(summary.num_edges, exp_num_edges)

    assert dataset.node_types
    assert dataset.edge_types

    num_nodes_per_type = {}
    for node_type in dataset.node_types:
        num_nodes_per_type[node_type] = torch.Tensor(
            [data.get_node_store(node_type).num_nodes for data in dataset])

    assert len(summary.num_nodes_per_type) == len(dataset.node_types)
    for node_type, node_type_stats in summary.num_nodes_per_type:
        check_summary_item(node_type_stats, num_nodes_per_type[node_type])

    num_edges_per_type = {}
    for edge_type in dataset.edge_types:
        num_edges_per_type[edge_type] = torch.Tensor(
            [data.get_edge_store(*edge_type).num_edges for data in dataset])

    assert len(summary.num_edges_per_type) == len(dataset.edge_types)
    for edge_type, edge_type_stats in summary.num_edges_per_type:
        check_summary_item(edge_type_stats, num_edges_per_type[edge_type])
