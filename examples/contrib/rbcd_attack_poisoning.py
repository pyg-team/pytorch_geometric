import copy
import os.path as osp
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rbcd_attack import GCN, metric, test, train
from torch import Tensor
from torch.optim import Adam

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import PRBCDAttack
from torch_geometric.datasets import Planetoid

try:
    import higher
except ImportError:
    sys.exit('Install `higher` via `pip install higher` for poisoning example')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: Edge weights are being ignored later and most adjacency matrix
# preprocessing should be part of the model (part of backpropagation):
dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

gcn = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
train(gcn, data)

print('------------- GCN: Global Poisoning -------------')

clean_acc = test(gcn, data)
print(f'Clean accuracy: {clean_acc:.3f}')

n_epochs = 50
lr = 0.04
weight_decay = 5e-4


class PoisoningPRBCDAttack(PRBCDAttack):
    def _forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                 **kwargs) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()

        with torch.enable_grad():
            ped = copy.copy(data)
            ped.x, ped.edge_index, ped.edge_weight = x, edge_index, edge_weight
            train(self.model, ped, n_epochs, lr, weight_decay)

        self.model.eval()
        return self.model(x, edge_index, edge_weight)

    def _forward_and_gradient(self, x: Tensor, labels: Tensor,
                              idx_attack: Optional[Tensor] = None,
                              **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()

        self.model.train()
        opt = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        with higher.innerloop_ctx(self.model, opt) as (fmodel, diffopt):
            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index, self.edge_weight, self.block_edge_index,
                self.block_edge_weight)

            # Normalize only once (only relevant if model normalizes adj)
            if hasattr(fmodel, 'norm'):
                edge_index, edge_weight = fmodel.norm(
                    edge_index,
                    edge_weight,
                    num_nodes=x.size(0),
                    add_self_loops=True,
                )

            for _ in range(n_epochs):
                pred = fmodel.forward(x, edge_index, edge_weight,
                                      skip_norm=True)
                loss = F.cross_entropy(pred[data.train_mask],
                                       data.y[data.train_mask])
                diffopt.step(loss)

            pred = fmodel(x, edge_index, edge_weight)
            loss = self.loss(pred, labels, idx_attack)

            gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Clip gradient for stability:
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        self.model.eval()

        return loss, gradient


prbcd = PoisoningPRBCDAttack(gcn, block_size=250_000, metric=metric, lr=100)

# PRBCD: Attack test set:
global_budget = int(0.05 * data.edge_index.size(1) / 2)  # Perturb 5% of edges

pert_edge_index, perts = prbcd.attack(
    data.x,
    data.edge_index,
    data.y,
    budget=global_budget,
    idx_attack=data.test_mask,
)

gcn.reset_parameters()
pert_data = copy.copy(data)
pert_data.edge_index = pert_edge_index
train(gcn, pert_data)
pert_acc = test(gcn, pert_data)
# Note that the values here a bit more noisy than in the evasion case:
print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {pert_acc:.3f}')

fig, ax1 = plt.subplots()
plt.title('Global Poisoning GCN')
color = 'tab:red'
ax1.plot(prbcd.attack_statistics['loss'], color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Steps')

# It is best practice choosing the learning rate s.t. the budget is exhausted:
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
