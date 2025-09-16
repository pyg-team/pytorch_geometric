import pytest
import torch

from torch_geometric.datasets import GraphLandDataset
from torch_geometric.testing import onlyOnline, withPackage


@onlyOnline
@withPackage('pandas', 'sklearn')
@pytest.mark.parametrize('name', [
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'tolokers-2',
    'city-reviews',
    'artnet-exp',
    'web-fraud',
    'hm-prices',
    'avazu-ctr',
    'city-roads-M',
    'city-roads-L',
    'twitch-views',
    'artnet-views',
    'web-traffic',
])
def test_transductive_graphland(name: str):
    dataset = GraphLandDataset(
        root='./datasets',
        split='RL',
        name=name,
        to_undirected=True,
    )
    assert len(dataset) == 1

    data = dataset[0]
    assert data.num_nodes == data.x.shape[0] == data.y.shape[0]

    assert not (data.train_mask & data.val_mask & data.test_mask).any().item()

    labeled_mask = data.train_mask | data.val_mask | data.test_mask
    assert not torch.isnan(data.y[labeled_mask]).any().item()
    assert not torch.isnan(data.x).any().item()

    assert not (data.x_numerical_mask & data.x_fraction_mask
                & data.x_categorical_mask).any().item()

    assert (data.x_numerical_mask | data.x_fraction_mask
            | data.x_categorical_mask).all().item()


@onlyOnline
@withPackage('pandas', 'sklearn')
@pytest.mark.parametrize('name', [
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'tolokers-2',
    'artnet-exp',
    'web-fraud',
    'hm-prices',
    'avazu-ctr',
    'twitch-views',
    'artnet-views',
])
def test_inductive_graphland(name: str):
    base_data = GraphLandDataset(
        root='./datasets',
        split='TH',
        name=name,
        to_undirected=True,
    )[0]
    num_nodes = base_data.num_nodes
    num_edges = base_data.num_edges
    del base_data

    dataset = GraphLandDataset(
        root='./datasets',
        split='THI',
        name=name,
        to_undirected=True,
    )
    assert len(dataset) == 3

    train_data, val_data, test_data = dataset
    assert num_nodes == test_data.num_nodes == test_data.node_id.shape[0]
    assert num_edges == test_data.num_edges

    assert not torch.isnan(train_data.y[train_data.mask]).any().item()
    assert not torch.isnan(val_data.y[val_data.mask]).any().item()
    assert not torch.isnan(test_data.y[test_data.mask]).any().item()
