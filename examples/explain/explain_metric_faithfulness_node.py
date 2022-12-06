import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metrics.faithfulness import (
    get_faithfulness,
    get_node_faithfulness,
)
from torch_geometric.nn import GCNConv

# based on examples/gnn_explainer.py
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(data.x, data.edge_index, data.edge_weight)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explainer_config=dict(
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
    ),
    model_config=dict(
        mode='classification',
        task_level='node',
        return_type='raw',
    ),
)

add_args = {'edge_weight': data.edge_weight}

topk = 10

for node_index in range(0, 50):
    explanation = explainer(data.x, data.edge_index, index=node_index,
                            edge_weight=data.edge_weight)
    _, node_feat_faithfulness = get_faithfulness(data, explanation, model,
                                                 topk, node_index=node_index,
                                                 **add_args)
    print(
        f'Faithfulness score with topk {topk} for node idx {node_index}  : {node_feat_faithfulness:.4f} '
    )
