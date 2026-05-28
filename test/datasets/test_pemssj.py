import pytest

from torch_geometric.datasets import PemsSJ
from torch_geometric.testing import onlyOnline


@onlyOnline
def test_pems():
    dataset = PemsSJ(root="./datasets/PemsSJ")

    assert str(dataset) == "PemsSJ(1)"
    assert len(dataset) == 1
    assert dataset.num_features == 1

    data = dataset[0]
    assert data.num_nodes == 1016
    assert data.edge_index.size() == (2, 2344)
    assert data.x.size() == (1016, 1)
    assert data.y.size() == (1016, 1)
    assert data.train_mask.size() == (1016, )
    assert data.test_mask.size() == (1016, )

    # Default split: no validation mask
    assert data.val_mask.sum().item() == 0

    # Masks are mutually exclusive
    overlap = (data.train_mask & data.test_mask).sum().item()
    assert overlap == 0

    assert dataset.original_mean.item() == pytest.approx(51.06)
    assert dataset.original_std.item() == pytest.approx(17.3341)


@onlyOnline
@pytest.mark.parametrize("val_pct", [0.1, 0.2])
def test_pems_validation_split(val_pct):
    dataset = PemsSJ(root="./datasets/PemsSJ", validation_percentage=val_pct)
    data = dataset[0]

    # Val mask carved from train: no three-way overlap
    assert (data.train_mask & data.test_mask).sum().item() == 0
    assert (data.train_mask & data.val_mask).sum().item() == 0
    assert (data.val_mask & data.test_mask).sum().item() == 0

    # Val size is roughly proportional
    n_train_default = 250
    num_val = data.val_mask.sum().item()
    assert num_val > 0
    assert int(n_train_default * val_pct) == num_val
