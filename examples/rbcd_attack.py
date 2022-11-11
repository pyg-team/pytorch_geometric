import os.path as osp
import functools
from typing import Optional, Tuple
import sys

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
# preprocessings should be part of the model (part of backpropagation)
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class GCN(torch.nn.Module):
    """GCN that normalizes adjacency matrix once for all layers and no caching."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        gcn_conv = functools.partial(
            GCNConv,
            cached=False,  # Important since we backpropagate to the adj. matrix
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

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
gat = GAT().to(device)
data = data.to(device)

x, edge_index, y = data.x, data.edge_index, data.y


def train(model: torch.nn.Module, x, edge_index, epochs=200,
          lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x, edge_index)
        loss = F.cross_entropy(
            prediction[data.train_mask], data.y[data.train_mask])
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
# The learning rate is one of the most important parameters for PRBCD and a
# good heuristic is to choose it s.t. the budget is exhausted within a few
# steps. Moreover, choose it high enough to lower the impact of the relaxation
# gap ({0, 1} -> [0, 1]) of the edge weights
lr_gcn = 1_000
lr_gat = 2_000

# The metric in PRBCD is assumed to be best if lower (i.e. like a loss)
metric = lambda *args, ** kwargs: -accuracy(*args, **kwargs)


print(f'\n------------- Local Evasion GCN -------------\n')
# Note: GRBCD is faster than PRBCD for small budgets but is not as consistent

grbcd = GRBCDAttack(gcn)
prbcd = PRBCDAttack(gcn, metric=metric, lr=lr_gcn)

clean_accuracy = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')

# GRBCD: attack single node
pert_edge_index, perts = grbcd.attack(
    x, edge_index, y, budget=local_budget, idx_attack=[node_idx])
clean_conf = F.softmax(gcn(x, edge_index)[node_idx], dim=-1)[y[node_idx]]
pert_conf = F.softmax(gcn(x, pert_edge_index)[
    node_idx], dim=-1)[y[node_idx]]
print(f'GRBCD: Confidence of target dropped from {clean_conf:.3f} '
      f'to {pert_conf:.3f}')
print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')

# PRBCD: attack single node
pert_edge_index, perts = prbcd.attack(
    x, edge_index, y, budget=local_budget, idx_attack=[node_idx])
clean_conf = F.softmax(gcn(x, edge_index)[node_idx], dim=-1)[y[node_idx]]
pert_conf = F.softmax(gcn(x, pert_edge_index)[
    node_idx], dim=-1)[y[node_idx]]
print(f'PRBCD: Confidence of target dropped from {clean_conf:.3f} '
      f'to {pert_conf:.3f}')
print(f'Adv. edges {", ".join(str((u, v)) for u, v in perts.T.tolist())}')


print(f'\n------------- Global Evasion GCN -------------\n')

grbcd = GRBCDAttack(gcn)
prbcd = PRBCDAttack(gcn, metric=metric, lr=lr_gcn)

clean_accuracy = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')

# GRBCD: attack test set
pert_edge_index, perts = grbcd.attack(
    x, edge_index, y, budget=global_budget, idx_attack=data.test_mask)

pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'GRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
      f'{pert_accuracy:.3f}')

# PRBCD: attack test set
pert_edge_index, perts = prbcd.attack(
    x, edge_index, y, budget=global_budget, idx_attack=data.test_mask)
pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'PRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
      f'{pert_accuracy:.3f}')

fig, ax1 = plt.subplots()
plt.title('Global Evasion GCN')
color = 'tab:red'
ax1.plot(prbcd.attack_statistics['loss'], color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Steps')

# It is best practice choosing the learning rate s.t. the budget is exhausted
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(prbcd.attack_statistics['prob_mass_after_update'],
         color=color, linestyle='--', label='Before projection')
ax2.plot(prbcd.attack_statistics['prob_mass_after_projection'],
         color=color, label='After projection')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Used budget')
plt.legend()
fig.show()
# TODO: delete
fig.savefig(f'plot_GCN.pdf')


try:
    import higher
except ImportError as e:
    sys.exit(
        'Install `higher` (e.g. `pip install higher`) for poisoning example')


print(f'\n------------- Global Poisoning GCN -------------\n')

clean_accuracy = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')


class PoisoningPRBCDAttack(PRBCDAttack):

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Tensor, **kwargs) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()
        self.model.train()

        with torch.enable_grad():
            train(self.model, x, edge_index, epochs=50, lr=0.04)

        self.model.eval()
        prediction = self.model(x, edge_index, edge_weight)

        return prediction

    def forward_and_gradient(self, x: Tensor, labels: Tensor,
                             idx_attack: Optional[Tensor] = None,
                             **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()
        # self.model.train()
        opt = torch.optim.Adam(
            self.model.parameters(), lr=0.04, weight_decay=5e-4)

        with higher.innerloop_ctx(self.model, opt) as (fmodel, diffopt):
            # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
            # (Algorithm 1, line 6 / Algorithm 2, line 7)
            edge_index, edge_weight = self._get_modified_adj()

            # Calculate loss combining all each node
            # (Algorithm 1, line 7 / Algorithm 2, line 8)
            # Includes in poisoning also the training.
            for _ in range(50):
                prediction = fmodel(x, edge_index, edge_weight)
                loss = F.cross_entropy(
                    prediction[data.train_mask], data.y[data.train_mask])
                diffopt.step(loss)

            prediction = fmodel(x, edge_index, edge_weight)
            loss = self.loss(prediction, labels, idx_attack)

            print(accuracy(prediction, labels, idx_attack))
            # Retrieve gradient towards the current block
            # (Algorithm 1, line 7 / Algorithm 2, line 8)
            gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Clip gradient for stability
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        # self.model.eval()

        return loss, gradient


prbcd = PoisoningPRBCDAttack(
    gcn, metric=metric, block_size=250_000, lr=40)

# PRBCD: attack test set
pert_edge_index, perts = prbcd.attack(
    x, edge_index, y, budget=global_budget, idx_attack=data.test_mask)

gcn.reset_parameters()
gcn.train()
train(gcn, x, pert_edge_index)
gcn.eval()
pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
print(f'PRBCD: Accuracy dropped from {clean_accuracy:.3f} to '
      f'{pert_accuracy:.3f}')

fig, ax1 = plt.subplots()
plt.title('Global Poisoning GCN')
color = 'tab:red'
ax1.plot(prbcd.attack_statistics['loss'], color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Steps')

# It is best practice choosing the learning rate s.t. the budget is exhausted
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(prbcd.attack_statistics['prob_mass_after_update'],
         color=color, linestyle='--', label='Before projection')
ax2.plot(prbcd.attack_statistics['prob_mass_after_projection'],
         color=color, label='After projection')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Used budget')
plt.legend()
fig.show()
# TODO: delete
fig.savefig(f'plot_pois_GCN.pdf')
