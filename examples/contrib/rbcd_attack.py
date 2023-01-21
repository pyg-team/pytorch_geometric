import copy
import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Normalize edge indices only once:
        if not kwargs.get('skip_norm', False):
            edge_index, edge_weight = self.norm(
                edge_index,
                edge_weight,
                num_nodes=x.size(0),
                add_self_loops=True,
            )

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges."""
    def edge_update(self, alpha_j: Tensor, alpha_i: Optional[Tensor],
                    edge_attr: Optional[Tensor], index: Tensor,
                    ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            assert edge_attr.dim() == 1, 'Only scalar edge weights supported'
            edge_attr = edge_attr.view(-1, 1)
            # `alpha` unchanged if edge_attr == 1 and -Inf if edge_attr == 0;
            # We choose log to counteract underflow in subsequent exp/softmax
            alpha = alpha + torch.log2(edge_attr)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Initialize edge weights of self-loops with 1:
        self.conv1 = WeightedGATConv(in_channels, hidden_channels,
                                     fill_value=1.)
        self.conv2 = WeightedGATConv(hidden_channels, out_channels,
                                     fill_value=1.)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight)
    return float(accuracy(pred, data.y, data.test_mask))


# The metric in PRBCD is assumed to be best if lower (like a loss).
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    gcn = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    gat = GAT(dataset.num_features, 16, dataset.num_classes).to(device)

    train(gcn, data)
    gcn.eval()

    train(gat, data)
    gat.eval()

    node_idx = 42
    local_budget = 2  # Degree of (training) node 42 is 2.

    # Perturb 5% of edges:
    global_budget = int(0.05 * data.edge_index.size(1) / 2)

    print('------------- GAT: Local Evasion -------------')
    # Note: GRBCD is faster than PRBCD for small budgets but not as consistent

    grbcd = GRBCDAttack(gat, block_size=250_000)
    # The learning rate is one of the most important parameters for PRBCD and a
    # good heuristic is to choose it s.t. the budget is exhausted within a few
    # steps. Moreover, a high learning rate mitigates the impact of the
    # relaxation gap ({0, 1} -> [0, 1]) of the edge weights. See poisoning
    #  example for a debug plot.
    prbcd = PRBCDAttack(gat, block_size=250_000, metric=metric, lr=2_000)

    clean_acc = test(gat, data)
    print(f'Clean accuracy: {clean_acc:.3f}')

    # GRBCD: Attack a single node:
    pert_edge_index, perts = grbcd.attack(
        data.x,
        data.edge_index,
        data.y,
        budget=local_budget,
        idx_attack=[node_idx],
    )

    clean_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, data.edge_index), data.y, [node_idx])
    pert_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, pert_edge_index), data.y, [node_idx])
    print(f'GRBCD: Confidence margin of target to best non-target dropped '
          f'from {clean_margin:.3f} to {pert_margin:.3f}')
    adv_edges = ', '.join(str((u, v)) for u, v in perts.T.tolist())
    print(f'Adv. edges: {adv_edges}')

    # PRBCD: Attack single node:
    pert_edge_index, perts = prbcd.attack(
        data.x,
        data.edge_index,
        data.y,
        budget=local_budget,
        idx_attack=[node_idx],
    )
    clean_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, data.edge_index), data.y, [node_idx])
    pert_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, pert_edge_index), data.y, [node_idx])
    print(f'PRBCD: Confidence margin of target to best non-target dropped '
          f'from {clean_margin:.3f} to {pert_margin:.3f}')
    adv_edges = ', '.join(str((u, v)) for u, v in perts.T.tolist())
    print(f'Adv. edges: {adv_edges}\n')

    print('------------- GCN: Global Evasion -------------')

    grbcd = GRBCDAttack(gcn, block_size=250_000)
    prbcd = PRBCDAttack(gcn, block_size=250_000, metric=metric, lr=2_000)

    clean_acc = test(gcn, data)

    # GRBCD: Attack test set:
    pert_edge_index, perts = grbcd.attack(
        data.x,
        data.edge_index,
        data.y,
        budget=global_budget,
        idx_attack=data.test_mask,
    )

    pert_data = copy.copy(data)
    pert_data.edge_index = pert_edge_index
    pert_acc = test(gcn, pert_data)
    print(f'GRBCD: Accuracy dropped from {clean_acc:.3f} to {pert_acc:.3f}')

    # PRBCD: Attack test set:
    pert_edge_index, perts = prbcd.attack(
        data.x,
        data.edge_index,
        data.y,
        budget=global_budget,
        idx_attack=data.test_mask,
    )

    pert_data = copy.copy(data)
    pert_data.edge_index = pert_edge_index
    pert_acc = test(gcn, pert_data)
    print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {pert_acc:.3f}')
