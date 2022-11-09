from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import (
    GCNConv, PRBCDAttack, GRBCDAttack, global_add_pool)
from torch_geometric.testing import is_full_test, withPackage


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x


attacks = [
    partial(PRBCDAttack, epochs_resampling=2, epochs_finetuning=2),
    GRBCDAttack
]


@pytest.mark.parametrize('attack', attacks)
@pytest.mark.parametrize('model', [GCN(), GNN()])
@pytest.mark.parametrize('budget', [1])
@pytest.mark.parametrize('is_undirected_graph', [False, True])
def test_attack(attack, model, budget, is_undirected_graph):
    attack = attack(model, log=False, is_undirected_graph=is_undirected_graph)
    assert attack.__repr__() in ['PRBCDAttack()', 'GRBCDAttack()']

    x = torch.randn(8, 3)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

    pert_edge_index, perturbations = attack(x, edge_index, y, budget)

    m = edge_index.size(1)
    if isinstance(attack, PRBCDAttack):
        assert perturbations.shape in [(2, 0), (2, 1)]

        if perturbations.size(1):
            if is_undirected_graph:
                possible_m = [m - 2, m + 2]
            else:
                possible_m = [m - 1, m + 1]
        else:
            possible_m = [m]
        assert pert_edge_index.size(1) in possible_m
    else:
        assert perturbations.shape == (2, 1)

        if is_undirected_graph:
            possible_m = [m - 2, m + 2]
        else:
            possible_m = [m - 1, m + 1]
        assert pert_edge_index.size(1) in possible_m

    if is_full_test():
        jit = torch.jit.export(attack)

        pert_edge_index, perturbations = jit.attack(x, edge_index, y, budget)
        if isinstance(attack, PRBCDAttack):
            assert perturbations.shape in [(2, 0), (2, 1)]

            if perturbations.size(1):
                if is_undirected_graph:
                    possible_m = [m - 2, m + 2]
                else:
                    possible_m = [m - 1, m + 1]
            else:
                possible_m = [m]
            assert pert_edge_index.size(1) in possible_m
        else:
            assert perturbations.shape == (2, 1)

            if is_undirected_graph:
                possible_m = [m - 2, m + 2]
            else:
                possible_m = [m - 1, m + 1]
            assert pert_edge_index.size(1) in possible_m
