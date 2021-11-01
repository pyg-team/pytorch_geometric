from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import BAShapes
from torch_geometric.nn import GCN, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

dataset = BAShapes(seed=0)
data = dataset[0]

model = GCN(data.num_node_features, 20, 3, dataset.num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
idx = np.arange(0, data.num_nodes)
train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)
data = data.to(device)
x, edge_index, y = data.x, data.edge_index, data.y

for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_idx], y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        train_correct = (out[train_idx].argmax(dim=1) == y[train_idx]).sum()
        train_acc = train_correct / len(train_idx)
        model.eval()
        out = model(x, edge_index)
        test_correct = (out[test_idx].argmax(dim=1) == y[test_idx]).sum()
        test_acc = test_correct / len(test_idx)
        print(f'Epoch: {epoch}, Train Accuracy: {train_acc}, ',
              'Test Accuracy: {test_acc}')

# Explain One Node
model.eval()
explainer = GNNExplainer(model, epochs=300, return_type='log_prob', log=False)
node_id = 400
_, edge_mask = explainer.explain_node(node_id, x, edge_index)
plt.figure(figsize=(20, 20))
ax, G = explainer.visualize_subgraph(node_id, edge_index, edge_mask, y=data.y)
plt.show()

# ROC AUC Over Test Nodes
targets = []
preds = []
for node_idx in tqdm(idx[data.test_mask]):
    _, edge_mask = explainer.explain_node(node_idx.item(), x, edge_index)
    subgraph = k_hop_subgraph(node_idx.item(), 3, edge_index)
    edge_label = torch.masked_select(data.edge_label, subgraph[3]).cpu()
    edge_mask = torch.masked_select(edge_mask, subgraph[3]).cpu()
    targets.extend(edge_label)
    preds.extend(edge_mask.cpu())

auc = roc_auc_score(targets, preds)
print(f'Mean ROC AUC: {auc}')
