import os.path as osp
import functools
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, GRBCDAttack, PRBCDAttack
from torch_geometric.utils import softmax

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# IMPORTANT: Edge weights are being ignored later and most adjacency
# preprocessings should be part of the model (we need to backprop through them)
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class GCN(torch.nn.Module):
    """GCN that normalizes adjacency matrix once for all layers and no caching."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        gcn_conv = functools.partial(
            GCNConv,
            cached=False,  # Important since we backprop to the adj. matrix
            add_self_loops=False,  # We add them in `self.gcn_norm`
            normalize=False  # It is better to normalize once
        )
        self.conv1 = gcn_conv(dataset.num_features, hidden_dim)
        self.conv2 = gcn_conv(hidden_dim, dataset.num_classes)
        self.gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm

    def forward(self, x, edge_index, edge_weight=None):
        # Normalizing once lowers memory footprint and caching is not possible
        # during attack (for training it would be possible)
        edge_index, edge_weight = self.gcn_norm(
            edge_index, edge_weight, x.size(0), add_self_loops=True)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
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
            # `alpha`` remains unchanged if edge_attr == 1 and is
            # -Inf if edge_attr == 0
            alpha = alpha + (1 - 1 / edge_attr)

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


def train(model: torch.nn.Module, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.cross_entropy(
            log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def accuracy(pred, lab, mask):
    return (pred.argmax(-1)[mask] == lab[mask]).float().mean()


train(gcn)
gcn.eval()
clean_accuracy_gcn = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Accuracy of gcn is {clean_accuracy_gcn:.3f}')
train(gat)
gat.eval()
clean_accuracy_gat = accuracy(gat(x, edge_index), y, data.test_mask).item()
print(f'Accuracy of gat is {clean_accuracy_gat:.3f}')


# To avoid excessive computation, we choose arbitrarily to analyze
# `GCN`` with `PRBCDAttack` and `GAT`` with `GRBCDAttack`
prbcd = PRBCDAttack(gcn, metric=lambda *args,
                    ** kwargs: -accuracy(*args, **kwargs))
grbcd = GRBCDAttack(gat, eps=5e-2)

# GRBCD: attack test set
pert_edge_index, perts = grbcd.attack(
    x, edge_index, y, budget=150, idx_attack=data.test_mask)

pert_accuracy = accuracy(gat(x, pert_edge_index), y, data.test_mask).item()
print(f'Accuracy dropped from {clean_accuracy_gat:.3f} to {pert_accuracy:.3f}')
print(f'Flipped edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

# GRBCD: attack single node
node_idx = 42
pert_edge_index, perts = grbcd.attack(
    x, edge_index, y, budget=2, idx_attack=[node_idx])
clean_conf = F.softmax(gat(x, edge_index)[node_idx], dim=-1)[y[node_idx]]
pert_conf = F.softmax(gat(x, pert_edge_index)[node_idx], dim=-1)[y[node_idx]]
print(f'Confidence of target dropped from {clean_conf:.3f} to {pert_conf:.3f}')
print(f'Flipped edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

# PRBCD: attack test set
pert_edge_index, perts = prbcd.attack(
    x, edge_index, y, budget=150, idx_attack=data.test_mask)
pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'Accuracy dropped from {clean_accuracy_gcn:.3f} to {pert_accuracy:.3f}')
print(f'Flipped edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(prbcd.attack_statistics['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Steps')

# It is best practice choosing the learning rate s.t. the budget is exhausted
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(prbcd.attack_statistics['prob_mass_after_update'],
         color=color, linestyle='--')
ax2.plot(prbcd.attack_statistics['prob_mass_after_projection'],
         color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Used budget')
fig.show()
# TODO: delete
fig.savefig('plot.pdf')
