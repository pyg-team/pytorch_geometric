"""
This is an example of using the PGM explainer algorithm
on a node classification task
"""
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.contrib.explain import PGMExplainer
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.nn import GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, normalize=False)
        self.conv2 = GCNConv(16, dataset.num_classes, normalize=False)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    x, edge_index, edge_weight, target = \
        data.x, data.edge_index, data.edge_weight, data.y

    model.train()
    for epoch in range(1, 500):
        optimizer.zero_grad()
        log_logits = model(x, edge_index, edge_weight)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    log_logits = model(x, edge_index, edge_weight)
    predicted_target = log_logits.argmax(dim=1)

    explainer = Explainer(
        model=model, algorithm=PGMExplainer(), node_mask_type='attributes',
        explanation_type='phenomenon',
        model_config=ModelConfig(mode='multiclass_classification',
                                 task_level='node', return_type='raw'))
    node_idx = 100
    explanation = explainer(x=data.x, edge_index=edge_index, index=node_idx,
                            target=predicted_target, edge_weight=edge_weight)
    print(f'Significance of relevant neighbors: {explanation.pgm_stats}')
