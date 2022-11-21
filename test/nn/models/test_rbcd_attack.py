import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import (
    GCNConv,
    GRBCDAttack,
    PRBCDAttack,
    global_add_pool,
)
from torch_geometric.testing import is_full_test
from torch_geometric.utils import to_undirected


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

    def forward(self, x, edge_index, edge_attr, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x


@pytest.mark.parametrize('attack_type', [PRBCDAttack, GRBCDAttack])
@pytest.mark.parametrize('model', [GCN, GNN])
@pytest.mark.parametrize('budget', [1])
@pytest.mark.parametrize('is_undirected_graph', [False, True])
def test_attack(attack_type, model, budget, is_undirected_graph):
    attack = attack_type(model(), epochs=4, epochs_resampling=2,
                         max_final_samples=2, log=False,
                         is_undirected_graph=is_undirected_graph)
    assert attack.__repr__() == f'{attack_type.__name__}()'

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
    if is_undirected_graph:
        edge_index = to_undirected(edge_index)

    if model == GNN:
        y = torch.tensor([0])
        # all nodes belong to same graph
        kwargs = dict(
            batch=torch.zeros(x.shape[0], dtype=int, device=x.device))
    else:
        y = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
        kwargs = {}

    pert_edge_index, perturbations = attack.attack(x, edge_index, y, budget,
                                                   **kwargs)

    m = edge_index.size(1)
    if budget == 1:
        if attack == PRBCDAttack:
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

        if budget == 1:
            if attack == PRBCDAttack:
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
