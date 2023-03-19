import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
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

for _ in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Select the output index of a node to be explained:
index = 10

# Instantiate the CaptumExplainer algorithm:
algorithm = CaptumExplainer(attribution_method='IntegratedGradients', )

# Create an instance of the Explainer class with the GNN model
# and the CaptumExplainer algorithm:
explainer = Explainer(
    model=model,
    algorithm=algorithm,
    explanation_type='model',  # or phenomenon
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=200,
    ),
)

# Compute the explanation:
explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    index=index,
)
print(f'Generated explanations in {explanation.available_explanations}')

# Visualize the node feature importance:
path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

# Visualize the subgraph:
path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph plot has been saved to '{path}'")
