import torch
from pytest import approx

from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.testing import withPackage


@withPackage('pandas')
@withPackage('tabulate')
def test_summary():
    seed_everything(0)
    dataset = FakeDataset(num_graphs=10)
    summary = dataset.summary()
    assert summary.num_graphs == 10
    num_nodes = torch.tensor([d.num_nodes for d in dataset]).float()
    assert summary.mean_num_nodes == approx(float(num_nodes.mean()))
    assert summary.std_num_nodes == approx(float(num_nodes.std()))
    assert summary.min_num_nodes == int(num_nodes.min())
    assert summary.max_num_nodes == int(num_nodes.max())

    num_edges = torch.tensor([d.num_edges for d in dataset]).float()
    assert summary.mean_num_edges == approx(float(num_edges.mean()))
    assert summary.std_num_edges == approx(float(num_edges.std()))
    assert summary.min_num_edges == int(num_edges.min())
    assert summary.max_num_edges == int(num_edges.max())


@withPackage('pandas')
@withPackage('tabulate')
def test_hetero_summary():
    seed_everything(0)
    dataset = FakeHeteroDataset(num_graphs=10)
    summary = dataset.summary()
    assert summary.num_graphs == 10
    num_nodes = torch.tensor([d.num_nodes for d in dataset]).float()
    assert summary.mean_num_nodes == approx(float(num_nodes.mean()))
    assert summary.std_num_nodes == approx(float(num_nodes.std()))
    assert summary.min_num_nodes == int(num_nodes.min())
    assert summary.max_num_nodes == int(num_nodes.max())

    num_edges = torch.tensor([d.num_edges for d in dataset]).float()
    assert summary.mean_num_edges == approx(float(num_edges.mean()))
    assert summary.std_num_edges == approx(float(num_edges.std()))
    assert summary.min_num_edges == int(num_edges.min())
    assert summary.max_num_edges == int(num_edges.max())
