import os.path as osp
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, GRBCDAttack, PRBCDAttack
from torch_geometric.utils import softmax

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges."""

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Optional[Tensor],
                edge_attr: Optional[Tensor], index: Tensor,
                ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            assert edge_attr.dim() == 1, 'Only scalar edge weights supported'
            edge_attr = edge_attr.view(-1, 1)
            # `alpha`` remains unchanged if edge_attr == 1 and is
            # -Inf if edge_attr == 0
            alpha = alpha + (1 - 1 / edge_attr)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GAT(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = WeightedGATConv(
            dataset.num_features, hidden_dim, fill_value=1.)
        self.conv2 = WeightedGATConv(
            hidden_dim, dataset.num_classes, fill_value=1.)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
gat = GAT().to(device)
data = data.to(device)

x, edge_index, y = data.x, data.edge_index, data.y


def train(model: torch.nn.Module):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


train(gcn)
train(gat)


def accuracy(pred, lab, mask):
    return (pred.argmax(-1)[mask] == lab[mask]).float().mean().item()


# To avoid excessive computation, we choose arbitrarily to analyze
# `GCN`` with `PRBCDAttack` and `GAT`` with `GRBCDAttack`
prbcd = PRBCDAttack(gcn)
grbcd = GRBCDAttack(gat, eps=5e-2)

# GRBCD: attack test set
pert_edge_index, perturbations = grbcd.attack(
    x, edge_index, y, budget=25, idx_attack=data.test_mask)

clean_accuracy = accuracy(gat(x, edge_index), y, data.test_mask)
pert_accuracy = accuracy(gat(x, pert_edge_index), y, data.test_mask)
print(f'Accuracy dropped from {clean_accuracy:.3f} to {pert_accuracy:.3f}')
print(f'Flipped edges {"".join(str(u, v) for u, v in perturbations.T)}')

# GRBCD: attack single node
node_idx = 42
pert_edge_index, perturbations = grbcd.attack(
    x, edge_index, y, budget=5, idx_attack=[node_idx])
clean_confidence = F.softmax(gat(x, edge_index)[node_idx], dim=-1)[y[node_idx]]
pert_confidence = F.softmax(
    gat(x, pert_edge_index)[node_idx], dim=-1)[y[node_idx]]
print(f'Confidenc dropped from {clean_accuracy:.3f} to {pert_accuracy:.3f}')
print(f'Flipped edges {"".join(str(u, v) for u, v in perturbations.T)}')

# PRBCD: attack test set
pert_edge_index, perturbations = prbcd.attack(
    x, edge_index, y, budget=25, idx_attack=data.test_mask)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(prbcd.attack_statistics.loss, color=color)
ax1.set_ylabel('Loss')

# It is best practice choosing the learning rate s.t. the budget is exhausted
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(prbcd.attack_statistics.pmass_update, color=color, linestyle='--')
ax2.plot(prbcd.attack_statistics.pmass_projected, color=color)
ax2.set_ylabel('Used budget')
