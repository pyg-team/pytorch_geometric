import functools
import os.path as osp
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import higher
except ImportError:
    sys.exit(
        'Install `higher` (e.g. `pip install higher`) for poisoning example')

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, PRBCDAttack

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

    def forward(self, x, edge_index, edge_weight=None, skip_norm=False):
        # Normalizing once lowers memory footprint and caching is not possible
        # during attack (for training it would be possible)
        if not skip_norm:
            edge_index, edge_weight = self.gcn_norm(edge_index, edge_weight,
                                                    x.size(0),
                                                    add_self_loops=True)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
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

print('\n------------- GCN: Global Poisoning -------------\n')

clean_accuracy = accuracy(gcn(x, edge_index), y, data.test_mask).item()
print(f'Clean accuracy: {clean_accuracy:.3f}')


class PoisoningPRBCDAttack(PRBCDAttack):
    def _forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                 **kwargs) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()
        self.model.train()

        with torch.enable_grad():
            train(self.model, x, edge_index, edge_weight=edge_weight,
                  epochs=50, lr=0.04)

        self.model.eval()
        prediction = self.model(x, edge_index, edge_weight)

        return prediction

    def _forward_and_gradient(self, x: Tensor, labels: Tensor,
                              idx_attack: Optional[Tensor] = None,
                              **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()
        # We purposely do not call `self.model.train()` to avoid dropout
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.04,
                               weight_decay=5e-4)

        with higher.innerloop_ctx(self.model, opt) as (fmodel, diffopt):
            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index, self.edge_weight, self.block_edge_index,
                self.block_edge_weight)

            # Normalize only once (only relevant if model normalizes adj.)
            if hasattr(fmodel, 'gcn_norm'):
                edge_index, edge_weight = fmodel.gcn_norm(
                    edge_index, edge_weight, x.size(0), add_self_loops=True)

            for _ in range(50):
                prediction = fmodel.forward(x, edge_index, edge_weight,
                                            skip_norm=True)
                loss = F.cross_entropy(prediction[data.train_mask],
                                       data.y[data.train_mask])
                diffopt.step(loss)

            prediction = fmodel(x, edge_index, edge_weight)
            loss = self.loss(prediction, labels, idx_attack)

            gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Clip gradient for stability
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        self.model.eval()

        return loss, gradient


# The metric in PRBCD is assumed to be best if lower (i.e. like a loss)
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


# for i in range(10):
prbcd = PoisoningPRBCDAttack(gcn, metric=metric, lr=100)

# PRBCD: attack test set
global_budget = int(0.05 * edge_index.size(1) / 2)  # Perturb 5% of edges
pert_edge_index, perts = prbcd.attack(x, edge_index, y, budget=global_budget,
                                      idx_attack=data.test_mask)

gcn.reset_parameters()
gcn.train()
train(gcn, x, pert_edge_index)
gcn.eval()
pert_accuracy = accuracy(gcn(x, pert_edge_index), y, data.test_mask).item()
# Note that the values here a bit more noisy than in the evasion case
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
ax2.plot(prbcd.attack_statistics['prob_mass_after_update'], color=color,
         linestyle='--', label='Before projection')
ax2.plot(prbcd.attack_statistics['prob_mass_after_projection'], color=color,
         label='After projection')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Used budget')
plt.legend()
fig.show()
