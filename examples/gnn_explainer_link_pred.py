import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import (
    Explainer,
    ExplainerConfig,
    GNNExplainer,
    ModelConfig,
)
from torch_geometric.nn import GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
])
dataset = Planetoid(path, dataset, transform=transform)
train_data, val_data, test_data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = model.encode(x, edge_index)
        return model.decode(z, edge_label_index).view(-1)


model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index,
                train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


for epoch in range(1, 11):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

# Explain model output for an edge
model_config = ModelConfig(mode="binary_classification", task_level="edge",
                           return_type="raw")
edge_label_index = val_data.edge_label_index[:, 0]
explainer = Explainer(
    model=model, algorithm=GNNExplainer(epochs=200),
    explainer_config=ExplainerConfig(
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
    ), model_config=model_config)
explanation = explainer(x=train_data.x, edge_index=train_data.edge_index,
                        edge_label_index=edge_label_index)
print(explanation.available_explanations)

# Explain a selected target (phenomenon) for an edge
explainer = Explainer(
    model=model, algorithm=GNNExplainer(epochs=200),
    explainer_config=ExplainerConfig(
        explanation_type="phenomenon",
        node_mask_type="attributes",
        edge_mask_type="object",
    ), model_config=model_config)
target = val_data.edge_label[0].unsqueeze(dim=0).long()
explanation = explainer(x=train_data.x, edge_index=train_data.edge_index,
                        target=target, edge_label_index=edge_label_index)
print(explanation.available_explanations)
