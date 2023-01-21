import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')


class GCN(torch.nn.Module):
    """GCN that normalizes adjacency matrix once for all layers (no cache)."""
    def __init__(self, num_features: int, num_classes: int,
                 hidden_dim: int = 16):
        super().__init__()
        # Important to backpropagate to the (input) adj. matrix. We normalize
        # the adj. once since it is expensive due to backpropagation.
        self.conv1 = GCNConv(num_features, hidden_dim, cached=False,
                             add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_dim, num_classes, cached=False,
                             add_self_loops=False, normalize=False)
        self.norm = gcn_norm

    def forward(self, x, edge_index, edge_weight=None, skip_norm=False):
        # Normalizing once lowers memory footprint
        if not skip_norm:
            edge_index, edge_weight = self.norm(edge_index, edge_weight,
                                                num_nodes=x.size(0),
                                                add_self_loops=True)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges (disabling edge features)."""
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
    """GAT that supports weights edges."""
    def __init__(self, num_features: int, num_classes: int,
                 hidden_dim: int = 16):
        super().__init__()
        # We add self-loops and initialize them to 1.
        self.conv1 = WeightedGATConv(num_features, hidden_dim,
                                     add_self_loops=True, fill_value=1.)
        self.conv2 = WeightedGATConv(hidden_dim, num_classes,
                                     add_self_loops=True, fill_value=1.)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model: torch.nn.Module, data: Data, epochs: int = 200,
          lr: float = 0.01, weight_decay: float = 5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(prediction[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def accuracy(pred, lab, mask):
    return (pred.argmax(-1)[mask] == lab[mask]).float().mean()


def test(model: torch.nn.Module, data: Data) -> float:
    model.eval()
    predictions = model(data.x, data.edge_index, data.edge_weight)
    return accuracy(predictions, data.y, data.test_mask).item()


if __name__ == '__main__':
    # IMPORTANT: Edge weights are being ignored later and most adjacency
    # preprocessings should be part of the model (part of backpropagation)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = GCN(dataset.num_features, dataset.num_classes).to(device)
    gat = GAT(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)

    train(gcn, data)
    gcn.eval()

    train(gat, data)
    gat.eval()

    local_budget = 2  # Degree of (training) node 42 is also 2
    # Perturb 5% of edges
    global_budget = int(0.05 * data.edge_index.size(1) / 2)
    node_idx = 42

    # The metric in PRBCD is assumed to be best if lower (like a loss).
    def metric(*args, **kwargs):
        return -accuracy(*args, **kwargs)

    print('\n------------- GAT: Local Evasion -------------\n')
    # Note: GRBCD is faster than PRBCD for small budgets but not as consistent

    grbcd = GRBCDAttack(gat, block_size=250_000)
    # The learning rate is one of the most important parameters for PRBCD and a
    # good heuristic is to choose it s.t. the budget is exhausted within a few
    # steps. Moreover, a high learning rate mitigates the impact of the
    # relaxation gap ({0, 1} -> [0, 1]) of the edge weights. See poisoning
    #  example for a debug plot.
    prbcd = PRBCDAttack(gat, block_size=250_000, metric=metric, lr=2_000)

    clean_accuracy = test(gat, data)
    print(f'Clean accuracy: {clean_accuracy:.3f}')

    # GRBCD: attack single node
    pert_edge_index, perts = grbcd.attack(data.x, data.edge_index, data.y,
                                          budget=local_budget,
                                          idx_attack=[node_idx])
    clean_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, data.edge_index), data.y, [node_idx])
    pert_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, pert_edge_index), data.y, [node_idx])
    print(
        f'GRBCD: Confidence margin of target to best non-target dropped from '
        f'{clean_margin:.3f} to {pert_margin:.3f}')
    print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

    # PRBCD: attack single node
    pert_edge_index, perts = prbcd.attack(data.x, data.edge_index, data.y,
                                          budget=local_budget,
                                          idx_attack=[node_idx])
    clean_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, data.edge_index), data.y, [node_idx])
    pert_margin = -PRBCDAttack._probability_margin_loss(
        gat(data.x, pert_edge_index), data.y, [node_idx])
    print(
        f'PRBCD: Confidence margin of target to best non-target dropped from '
        f'{clean_margin:.3f} to {pert_margin:.3f}')
    print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

    print('\n------------- GCN: Global Evasion -------------\n')

    grbcd = GRBCDAttack(gcn, block_size=250_000)
    prbcd = PRBCDAttack(gcn, block_size=250_000, metric=metric, lr=2_000)

    clean_accuracy = test(gcn, data)
    print(f'Clean accuracy: {clean_accuracy:.3f}')

    # GRBCD: attack test set
    pert_edge_index, perts = grbcd.attack(data.x, data.edge_index, data.y,
                                          budget=global_budget,
                                          idx_attack=data.test_mask)

    pert_data = data.clone()
    pert_data.edge_index = pert_edge_index
    pert_accuracy = test(gcn, pert_data)
    print(f'GRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
          f'{pert_accuracy:.3f}')

    # PRBCD: attack test set
    pert_edge_index, perts = prbcd.attack(data.x, data.edge_index, data.y,
                                          budget=global_budget,
                                          idx_attack=data.test_mask)
    pert_data.edge_index = pert_edge_index
    pert_accuracy = test(gcn, pert_data)
    print(f'PRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
          f'{pert_accuracy:.3f}')
