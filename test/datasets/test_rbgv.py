import pytest
import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import RBGVDataset
from torch_geometric.utils import is_undirected


def test_rbgv_dataset():
    seed_everything(12345)
    dataset = RBGVDataset(num_graphs=10)

    assert str(dataset) == ('RBGVDataset(10, topology=erdos_renyi, '
                            'spurious_target=none, spurious_strategy=all)')
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

        # Without spurious edges, confounders stay isolated -> all relevant:
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
@pytest.mark.parametrize('target', ['red', 'blue', 'both'])
@pytest.mark.parametrize('strategy', ['all', 'normal'])
def test_rbgv_dataset_spurious(topology, target, strategy):
    seed_everything(12345)
    dataset = RBGVDataset(
        num_graphs=10,
        topology=topology,
        spurious_target=target,
        spurious_strategy=strategy,
    )

    for data in dataset:
        data.validate(raise_on_error=True)
        num_nodes = data.num_nodes

        # A spurious edge is exactly an edge touching a confounder:
        is_spurious = data.edge_mask < 0.5
        touches_confounder = (data.edge_index >= num_nodes - 2).any(dim=0)
        assert torch.equal(is_spurious, touches_confounder)

        assert is_undirected(data.edge_index)


def test_rbgv_dataset_asymmetric_dict():
    # Per-confounder dicts: green densely connects to every red node, violet
    # densely to every blue node, fully independently.
    seed_everything(12345)
    dataset = RBGVDataset(
        num_graphs=10,
        spurious_target={
            'green': 'red',
            'violet': 'blue'
        },
        spurious_strategy={
            'green': 'all',
            'violet': 'all'
        },
    )

    assert str(dataset) == (
        "RBGVDataset(10, topology=erdos_renyi, "
        "spurious_target={'green': 'red', 'violet': 'blue'}, "
        "spurious_strategy={'green': 'all', 'violet': 'all'})")

    for data in dataset:
        colors = data.x.argmax(dim=1)
        num_nodes = data.num_nodes
        green_idx, violet_idx = num_nodes - 2, num_nodes - 1
        row, col = data.edge_index

        # Green is connected to *all* and *only* red nodes:
        green_neighbors = col[row == green_idx]
        assert bool((colors[green_neighbors] == 0).all())
        assert int(green_neighbors.numel()) == int((colors == 0).sum())
        # Violet is connected to *all* and *only* blue nodes:
        violet_neighbors = col[row == violet_idx]
        assert bool((colors[violet_neighbors] == 1).all())
        assert int(violet_neighbors.numel()) == int((colors == 1).sum())


def test_rbgv_dataset_mixed_strategies():
    # Green dense over red, violet stochastic over blue -> still a valid graph
    # whose spurious edges are exactly the confounder-touching ones.
    seed_everything(12345)
    dataset = RBGVDataset(
        num_graphs=10,
        spurious_target={
            'green': 'red',
            'violet': 'blue'
        },
        spurious_strategy={
            'green': 'all',
            'violet': 'normal'
        },
    )

    for data in dataset:
        data.validate(raise_on_error=True)
        num_nodes = data.num_nodes
        is_spurious = data.edge_mask < 0.5
        touches_confounder = (data.edge_index >= num_nodes - 2).any(dim=0)
        assert torch.equal(is_spurious, touches_confounder)


def test_rbgv_dataset_string_equals_symmetric_dict():
    # A single string must behave exactly like the equivalent symmetric dict.
    seed_everything(12345)
    dataset1 = RBGVDataset(num_graphs=5, spurious_target='both',
                           spurious_strategy='normal')

    seed_everything(12345)
    dataset2 = RBGVDataset(
        num_graphs=5,
        spurious_target={
            'green': 'both',
            'violet': 'both'
        },
        spurious_strategy={
            'green': 'normal',
            'violet': 'normal'
        },
    )

    for data1, data2 in zip(dataset1, dataset2):
        assert torch.equal(data1.edge_index, data2.edge_index)


def test_rbgv_dataset_invalid_args():
    with pytest.raises(ValueError, match="node range"):
        RBGVDataset(num_graphs=1, min_nodes=10, max_nodes=5)
    with pytest.raises(ValueError, match="topology"):
        RBGVDataset(num_graphs=1, topology='unknown')
    with pytest.raises(ValueError, match="spurious_target"):
        RBGVDataset(num_graphs=1, spurious_target='yellow')
    with pytest.raises(ValueError, match="spurious_strategy"):
        RBGVDataset(num_graphs=1, spurious_strategy='dense')
    # Dictionary with the wrong keys:
    with pytest.raises(ValueError, match="exactly the keys"):
        RBGVDataset(num_graphs=1, spurious_target={'green': 'red'})
    # Per-confounder dict value out of range:
    with pytest.raises(ValueError, match="violet confounder"):
        RBGVDataset(num_graphs=1, spurious_target={
            'green': 'red',
            'violet': 'yellow'
        })
    # Wrong type:
    with pytest.raises(TypeError, match="must be a string or a dictionary"):
        RBGVDataset(num_graphs=1, spurious_target=42)


def test_rbgv_dataset_reproducibility():
    seed_everything(12345)
    dataset1 = RBGVDataset(
        num_graphs=10,
        spurious_target={
            'green': 'red',
            'violet': 'both'
        },
        spurious_strategy='normal',
    )

    seed_everything(12345)
    dataset2 = RBGVDataset(
        num_graphs=10,
        spurious_target={
            'green': 'red',
            'violet': 'both'
        },
        spurious_strategy='normal',
    )

    for data1, data2 in zip(dataset1, dataset2):
        assert torch.equal(data1.edge_index, data2.edge_index)
        assert torch.equal(data1.edge_mask, data2.edge_mask)
        assert torch.equal(data1.y, data2.y)
