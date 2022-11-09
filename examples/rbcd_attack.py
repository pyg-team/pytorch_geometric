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

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


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
            # Alpha remains unchanged if edge_attr == 1 and is
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

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
gat = GAT().to(device)
data = data.to(device)


def train(model: torch.nn.Module):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

    for _ in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index, edge_weight)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


train(gcn)
train(gat)

explainer = GNNExplainer(gcn, epochs=200, return_type='log_prob')
node_idx = 10
node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index,
                                                   edge_weight=edge_weight)




