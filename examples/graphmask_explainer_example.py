import os.path as osp

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from torch_geometric.explain import Explainer, GraphMaskExplainer
from torch_geometric.nn import (
    GCNConv, GATConv, FastRGCNConv, global_add_pool, global_mean_pool)
import torch_geometric.transforms as T
from torch_geometric.datasets import Entities, TUDataset, Planetoid
from torch_geometric.loader import DataLoader

# GCN Node Classification=================================================
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, add_self_loops=False)
        self.conv2 = GCNConv(16, dataset.num_classes, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(data.x, data.edge_index)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=5),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='probs',
    ),
)
node_index = 10
explanation = explainer(data.x, data.edge_index, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='graphviz')
print(f"Subgraph visualization plot has been saved to '{path}'")


# GAT Node Classification======================================================
transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8)
        self.conv2 = GATConv(64, dataset.num_classes, heads=1,
                             concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)
x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

for epoch in range(1, 6):
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print('Epoch {:03d}'.format(epoch),
          'train_loss: {:.4f}'.format(loss))
print("Optimization Finished!")

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=5, layer_type='GAT'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
)
node_index = torch.Tensor([10, 20]).long()
explanation = explainer(data.x, data.edge_index, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='networkx')
print(f"Subgraph visualization plot has been saved to '{path}'")


# RGCN Node Classification====================================================
dataset = Entities(root='/tmp/AIFB', name='AIFB')
edge_index, edge_type, train_labels, test_labels, train_mask, test_mask, \
    num_nodes = dataset[0].edge_index, dataset[0].edge_type, \
    dataset[0].train_y, dataset[0].test_y, \
    dataset[0].train_idx, dataset[0].test_idx, dataset[0].num_nodes
x = torch.randn(8285, 4)
edge_y = torch.randint(low=0, high=30, size=(edge_index.size(1),))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastRGCNConv(4, 8, 90, num_blocks=2, aggr='mean')
        self.conv2 = FastRGCNConv(8, 12, 90, num_blocks=4, aggr='add')
        self.conv3 = FastRGCNConv(12, 4, 90, num_blocks=2, aggr='mean')

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)

for epoch in range(1, 5):
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index, edge_type)
    loss = F.nll_loss(log_logits[train_mask], train_labels)
    loss.backward()
    optimizer.step()
    print('Epoch {:03d}'.format(epoch),
          'train_loss: {:.4f}'.format(loss))
print("Optimization Finished!")

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(3, epochs=5, layer_type='FastRGCN'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node',
        return_type='raw',
    ),
)
node_index = torch.Tensor([20, 10]).long()
explanation = explainer(x, edge_index, index=node_index, edge_type=edge_type)

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='networkx')
print(f"Subgraph visualization plot has been saved to '{path}'")


# GAT Link Prediction==========================================================
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
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(64, out_channels, heads=1,
                             concat=False)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

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
    if epoch % 20 == 0:
        val_auc = test(val_data)
        test_auc = test(test_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

# Explain model output for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]

explainer = Explainer(
    model=model,
    explanation_type='model',
    algorithm=GraphMaskExplainer(2, epochs=5, layer_type='GAT'),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='edge',
        return_type='raw',
    ),
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    edge_label_index=edge_label_index,
)
print(f'Generated model explanations in {explanation.available_explanations}')

# Explain a selected target (phenomenon) for a single edge:
edge_label_index = val_data.edge_label_index[:, 0]
target = val_data.edge_label[0].unsqueeze(dim=0).long()

explainer = Explainer(
    model=model,
    explanation_type='phenomenon',
    algorithm=GraphMaskExplainer(2, epochs=5, layer_type='GAT'),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='edge',
        return_type='raw',
    ),
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    target=target,
    edge_label_index=edge_label_index,
)
available_explanations = explanation.available_explanations
print(f'Generated phenomenon explanations in {available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='networkx')
print(f"Subgraph visualization plot has been saved to '{path}'")


# GCN Graph Classification=====================================================
dataset = TUDataset(root='ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=100, shuffle=True)
edge_y = torch.randint(low=0, high=30,
                       size=(dataset[4].edge_index.size(1),))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 7, add_self_loops=True, normalize=False)
        self.conv2 = GCNConv(7, 9, add_self_loops=False, normalize=True)
        self.conv3 = GCNConv(9, 6, add_self_loops=True, normalize=False)

    def forward(self, x, edge_index, batch, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)

for epoch in range(1, 5):
    loss_all = 0
    for data in loader:
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch,
                       data.edge_weight)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    print('Epoch {:03d}'.format(epoch),
          'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
print("Optimization Finished!")

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(3, epochs=5),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='raw',
    ),
)
graph_index = 10
edge_weight = torch.rand(dataset[graph_index].edge_index.size(1))
batch = torch.zeros(dataset[graph_index].x.shape[0],
                    dtype=int, device=dataset[graph_index].x.device)
explanation = explainer(dataset[graph_index].x,
                        dataset[graph_index].edge_index,
                        batch=batch, edge_weight=edge_weight,
                        index=graph_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='networkx')
print(f"Subgraph visualization plot has been saved to '{path}'")


# GAT Graph Classification=====================================================
dataset = TUDataset(root='ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=100, shuffle=True)
edge_y = torch.randint(low=0, high=30,
                       size=(dataset[4].edge_index.size(1),))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 8, heads=8, add_self_loops=True)
        self.conv2 = GATConv(64, 6, heads=1, concat=False,
                             add_self_loops=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)

for epoch in range(1, 5):
    loss_all = 0
    for data in loader:
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    print('Epoch {:03d}'.format(epoch),
          'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
print("Optimization Finished!")

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=5, layer_type='GAT'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='common_attributes',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='probs',
    ),
)
graph_index = 10
batch = torch.zeros(dataset[graph_index].x.shape[0],
                    dtype=int, device=dataset[graph_index].x.device)
explanation = explainer(dataset[graph_index].x,
                        dataset[graph_index].edge_index,
                        batch=batch, index=graph_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='graphviz')
print(f"Subgraph visualization plot has been saved to '{path}'")


# RGCN Graph Classification===================================================
dataset = TUDataset(root='ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=100, shuffle=True)
edge_type = torch.randint(low=0, high=90,
                          size=(dataset[4].edge_index.size(1),))
edge_y = torch.randint(low=0, high=30,
                       size=(dataset[4].edge_index.size(1),))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastRGCNConv(3, 6, 90, num_blocks=3, aggr='mean')
        self.conv2 = FastRGCNConv(6, 3, 90, num_blocks=3, aggr='add')
        self.conv3 = FastRGCNConv(3, 6, 90, num_blocks=3, aggr='mean')

    def forward(self, x, edge_index, batch, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)

for epoch in range(1, 5):
    loss_all = 0
    for data in loader:
        model.train()
        optimizer.zero_grad()
        data.edge_type = torch.randint(
            low=0, high=90, size=(data.edge_index.size(1),))
        output = model(data.x, data.edge_index, data.batch, data.edge_type)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    print('Epoch {:03d}'.format(epoch),
          'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
print("Optimization Finished!")

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(3, epochs=5, layer_type='FastRGCN'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='raw',
    ),
)
graph_index = 10
batch = torch.zeros(dataset[graph_index].x.shape[0],
                    dtype=int, device=dataset[graph_index].x.device)
edge_type = torch.randint(low=0, high=90,
                          size=(dataset[graph_index].edge_index.size(1),))
explanation = explainer(dataset[graph_index].x,
                        dataset[graph_index].edge_index,
                        edge_type=edge_type, batch=batch, index=graph_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path, backend='networkx')
print(f"Subgraph visualization plot has been saved to '{path}'")
