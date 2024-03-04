import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.ba2motif_dataset import BA2MotifDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv

from torch_geometric.explain.metric import fidelity
from tools.fidelity_pyg import robust_fidelity
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ba2')
dataset = BA2MotifDataset(path)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(20 * 2, 2)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(x.device)
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset.data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 20):
    model.train()
    optimizer.zero_grad()
    out = model(dataset[0].x, dataset[0].edge_index)
    loss = F.nll_loss(out, dataset[0].y)
    loss.backward()
    optimizer.step()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)


explanation = explainer(dataset[0].x, dataset[0].edge_index)
print(f'Generated explanations in {explanation.available_explanations}')

# path = 'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.pdf'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")

f_plus,f_minus = fidelity(explainer,explanation)

print("finished")

f_plus,f_minus,f_d, f_plus_label,f_minus_label,f_d_label = robust_fidelity(explainer,explanation,undirect=True,top_k=5)

print("node robust fidelity finished")

# exit(0)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class GCN(torch.nn.Module):
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
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)
node_index = 10
explanation = explainer(data.x, data.edge_index, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

# path = 'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.pdf'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")

f_plus,f_minus = fidelity(explainer,explanation)

print("finished")

f_plus,f_minus,f_d, f_plus_label,f_minus_label,f_d_label = robust_fidelity(explainer,explanation,undirect=False,top_k=5)

print("node robust fidelity finished")

