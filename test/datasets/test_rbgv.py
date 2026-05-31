import pytest
import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import RBGVDataset
from torch_geometric.utils import is_undirected


def test_rbgv_dataset():
    seed_everything(12345)
    dataset = RBGVDataset(num_graphs=10)

    assert str(dataset) == ('RBGVDataset(10, topology=erdos_renyi, '
                            'leakage_target=none, leakage_strategy=all)')
    assert len(dataset) == 10

    for data in dataset:
        data.validate(raise_on_error=True)
        num_nodes = data.num_nodes

        # Exactly two confounders (one green, one violet) per graph:
        assert num_nodes >= 5 + 2 and num_nodes <= 15 + 2
        assert data.x.size() == (num_nodes, 4)
        # Features are a valid one-hot encoding:
        assert torch.equal(data.x.sum(dim=1), torch.ones(num_nodes))
        assert int(data.x[:, 2].sum()) == 1  # One green node.
        assert int(data.x[:, 3].sum()) == 1  # One violet node.

        # Graph-level binary label:
        assert data.y.size() == (1, )
        assert data.y.item() in (0, 1)

        # Ground-truth masks have the right shape and value range:
        assert data.node_mask.size() == (num_nodes, 1)
        assert data.edge_mask.size() == (data.num_edges, )
        assert data.node_mask.min() == 0 and data.node_mask.max() == 1

        # Causally relevant nodes are exactly the red/blue ones:
        assert int(data.node_mask.sum()) == num_nodes - 2

        # Without leakage, the confounders stay isolated -> all edges relevant:
        assert int(data.edge_mask.sum()) == data.num_edges

        assert is_undirected(data.edge_index)


def test_rbgv_dataset_label():
    seed_everything(12345)
    dataset = RBGVDataset(num_graphs=20)

    for data in dataset:
        colors = data.x.argmax(dim=1)
        num_red = int((colors == 0).sum())
        num_blue = int((colors == 1).sum())
        # Label is 1 iff blue strictly outnumbers red:
        assert data.y.item() == int(num_blue > num_red)


@pytest.mark.parametrize('topology', ['erdos_renyi', 'barabasi_albert'])
@pytest.mark.parametrize('leakage_target', ['red', 'blue', 'both'])
@pytest.mark.parametrize('leakage_strategy', ['all', 'normal'])
def test_rbgv_dataset_leakage(topology, leakage_target, leakage_strategy):
    seed_everything(12345)
    dataset = RBGVDataset(
        num_graphs=10,
        topology=topology,
        leakage_target=leakage_target,
        leakage_strategy=leakage_strategy,
    )

    for data in dataset:
        data.validate(raise_on_error=True)
        num_nodes = data.num_nodes

        # A leakage edge is exactly an edge touching a confounder:
        is_leakage = data.edge_mask < 0.5
        touches_confounder = (data.edge_index >= num_nodes - 2).any(dim=0)
        assert torch.equal(is_leakage, touches_confounder)

        assert is_undirected(data.edge_index)


def test_rbgv_dataset_all_leakage_targets():
    # The `'all'` strategy connects every confounder to every target node, so
    # the (undirected) leakage edge count is deterministic given the colors.
    seed_everything(12345)
    dataset = RBGVDataset(num_graphs=10, leakage_target='both',
                          leakage_strategy='all')

    for data in dataset:
        num_nodes = data.num_nodes
        num_main = num_nodes - 2
        num_leakage = int((data.edge_mask < 0.5).sum())
        # Two confounders, each wired to all `num_main` nodes, counted in both
        # directions: `2 * num_main * 2`.
        assert num_leakage == 2 * num_main * 2
        # Both relevant (1) and leakage (0) edges are present:
        assert data.edge_mask.min() == 0 and data.edge_mask.max() == 1


def test_rbgv_dataset_invalid_args():
    with pytest.raises(ValueError, match="node range"):
        RBGVDataset(num_graphs=1, min_nodes=10, max_nodes=5)
    with pytest.raises(ValueError, match="topology"):
        RBGVDataset(num_graphs=1, topology='unknown')
    with pytest.raises(ValueError, match="leakage target"):
        RBGVDataset(num_graphs=1, leakage_target='yellow')
    with pytest.raises(ValueError, match="leakage strategy"):
        RBGVDataset(num_graphs=1, leakage_strategy='dense')


def test_rbgv_dataset_reproducibility():
    seed_everything(12345)
    dataset1 = RBGVDataset(num_graphs=10, leakage_target='both',
                           leakage_strategy='normal')

    seed_everything(12345)
    dataset2 = RBGVDataset(num_graphs=10, leakage_target='both',
                           leakage_strategy='normal')

    for data1, data2 in zip(dataset1, dataset2):
        assert torch.equal(data1.edge_index, data2.edge_index)
        assert torch.equal(data1.edge_mask, data2.edge_mask)
        assert torch.equal(data1.y, data2.y)
