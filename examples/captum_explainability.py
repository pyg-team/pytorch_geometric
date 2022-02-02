import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer
from torch_geometric.nn.models.explainer import to_captum

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
transform = T.NormalizeFeatures()
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, edge_index = data.x, data.edge_index

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

node_idx = 10
captum_model = to_captum(model, mask_type="edge", node_idx=node_idx).to(device)
target = data.y[node_idx].item()
input_mask = torch.ones(data.num_edges,
                        dtype=torch.float).requires_grad_(True).to(device)

# Edge explainability with Integrated Gradients for node 10
ig = IntegratedGradients(captum_model)
ig_attr = ig.attribute(input_mask.unsqueeze(0), target=target,
                       additional_forward_args=(x, edge_index),
                       internal_batch_size=1)

# Edge explainability with Saliency for node 10
sa = Saliency(captum_model)
sa_attr = sa.attribute(input_mask.unsqueeze(0), target=target,
                       additional_forward_args=(x, edge_index))

# Scale attributions to [0, 1]
ig_attr = ig_attr.squeeze(0).abs() / ig_attr.abs().max()
sa_attr = sa_attr.squeeze(0).abs() / sa_attr.abs().max()

# Visualize attributions with GNNExplainer visualizer
explainer = GNNExplainer(model)
ax, G = explainer.visualize_subgraph(node_idx, edge_index, ig_attr, y=data.y)
plt.show()
ax, G = explainer.visualize_subgraph(node_idx, edge_index, sa_attr, y=data.y)
plt.show()
