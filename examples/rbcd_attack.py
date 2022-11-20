import functools
import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv, GRBCDAttack, PRBCDAttack
from torch_geometric.utils import softmax

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# IMPORTANT: Edge weights are being ignored later and most adjacency
# preprocessings should be part of the model (part of backpropagation)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GCN(torch.nn.Module):
    """GCN that normalizes adjacency matrix once for all layers (no cache)."""
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        gcn_conv = functools.partial(
            GCNConv,
            cached=False,  # Important to backpropagate to the adj. matrix
            add_self_loops=False,  # We add them in `self.gcn_norm`
            normalize=False  # It is better to normalize once
        )
        self.conv1 = gcn_conv(dataset.num_features, hidden_dim)
        self.conv2 = gcn_conv(hidden_dim, dataset.num_classes)
        self.gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm

    def forward(self, x, edge_index, edge_weight=None):
        # Normalizing once lowers memory footprint and caching is not possible
        # during attack (for training it would be possible)
        edge_index, edge_weight = self.gcn_norm(edge_index, edge_weight,
                                                x.size(0), add_self_loops=True)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


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
            # `alpha`` unchanged if edge_attr == 1 and -Inf if edge_attr == 0;
            # We choose log to counteract underflow in subsequent exp/softmax
            alpha = alpha + torch.log2(edge_attr)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GAT(torch.nn.Module):
    """GAT that supports weights edges."""
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        gat_conv = functools.partial(
            WeightedGATConv,
            add_self_loops=True,  # In case the dataset has none
            fill_value=1.,  # Default edge weight (for self-loops)
        )
        self.conv1 = gat_conv(dataset.num_features, hidden_dim)
        self.conv2 = gat_conv(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
gat = GAT().to(device)
data = data.to(device)

x, edge_index, y = data.x, data.edge_index, data.y


def train(model, x, edge_index, edge_weight=None, epochs=200, lr=0.01,
          weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x, edge_index, edge_weight)
        loss = F.cross_entropy(prediction[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def accuracy(pred, lab, mask):
    return (pred.argmax(-1)[mask] == lab[mask]).float().mean()


gcn.train()
train(gcn, x, edge_index)
gcn.eval()

gat.train()
train(gat, x, edge_index)
gat.eval()

local_budget = 2  # Degree of (training) node 42 is also 2
global_budget = int(0.05 * edge_index.size(1) / 2)  # Perturb 5% of edges
node_idx = 42


# The metric in PRBCD is assumed to be best if lower (like a loss).
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


print('\n------------- GAT: Local Evasion -------------\n')
# Note: GRBCD is faster than PRBCD for small budgets but is not as consistent

grbcd = GRBCDAttack(gat)
# The learning rate is one of the most important parameters for PRBCD and a
# good heuristic is to choose it s.t. the budget is exhausted within a few
# steps. Moreover, a high learning rate mitigates the impact of the relaxation
# gap ({0, 1} -> [0, 1]) of the edge weights. See poisoning example for a plot.
prbcd = PRBCDAttack(gat, metric=metric, lr=2_000)

clean_accuracy = accuracy(gat(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')

# GRBCD: attack single node
pert_edge_index, perts = grbcd.attack(x, edge_index, y, budget=local_budget,
                                      idx_attack=[node_idx])
clean_margin = -PRBCDAttack._probability_margin_loss(gat(x, edge_index), y,
                                                     [node_idx])
pert_margin = -PRBCDAttack._probability_margin_loss(gat(x, pert_edge_index), y,
                                                    [node_idx])
print(f'GRBCD: Confidence margin of target to best non-target dropped from '
      f'{clean_margin:.3f} to {pert_margin:.3f}')
print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

# PRBCD: attack single node
pert_edge_index, perts = prbcd.attack(x, edge_index, y, budget=local_budget,
                                      idx_attack=[node_idx])
clean_margin = -PRBCDAttack._probability_margin_loss(gat(x, edge_index), y,
                                                     [node_idx])
pert_margin = -PRBCDAttack._probability_margin_loss(gat(x, pert_edge_index), y,
                                                    [node_idx])
print(f'PRBCD: Confidence margin of target to best non-target dropped from '
      f'{clean_margin:.3f} to {pert_margin:.3f}')
print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

print('\n------------- GCN: Global Evasion -------------\n')

grbcd = GRBCDAttack(gcn)
prbcd = PRBCDAttack(gcn, metric=metric, lr=2_000)

clean_accuracy = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')

# GRBCD: attack test set
pert_edge_index, perts = grbcd.attack(x, edge_index, y, budget=global_budget,
                                      idx_attack=data.test_mask)

pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'GRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
      f'{pert_accuracy:.3f}')

# PRBCD: attack test set
pert_edge_index, perts = prbcd.attack(x, edge_index, y, budget=global_budget,
                                      idx_attack=data.test_mask)
pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'PRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
      f'{pert_accuracy:.3f}')
