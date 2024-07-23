import pytest
import torch

from torch_geometric.datasets import KarateClub
from torch_geometric.testing import withPackage
from torch_geometric.utils import get_ppr


@withPackage('numba')
@pytest.mark.parametrize('target', [None, torch.tensor([0, 4, 5, 6])])
def test_get_ppr(target):
    data = KarateClub()[0]

    edge_index, edge_weight = get_ppr(
        data.edge_index,
        alpha=0.1,
        eps=1e-5,
        target=target,
    )

    assert edge_index.size(0) == 2
    assert edge_index.size(1) == edge_weight.numel()

    min_row = 0 if target is None else target.min()
    max_row = data.num_nodes - 1 if target is None else target.max()
    assert edge_index[0].min() == min_row and edge_index[0].max() == max_row
    assert edge_index[1].min() >= 0 and edge_index[1].max() < data.num_nodes
    assert edge_weight.min() >= 0.0 and edge_weight.max() <= 1.0
