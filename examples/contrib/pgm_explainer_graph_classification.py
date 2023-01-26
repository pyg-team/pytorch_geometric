"""
This is an example of using the PGM explainer algorithm
on a graph classification task
"""
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.contrib.explain import PGMExplainer
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.explain import Explainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.utils import normalized_cut

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
transform = T.Cartesian(cat=False)
train_dataset = MNISTSuperpixels(path, True, transform=transform)
test_dataset = MNISTSuperpixels(path, False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
d = train_dataset


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(),
                            nn.Linear(25, d.num_features * 32))
        self.conv1 = NNConv(d.num_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(),
                            nn.Linear(25, 32 * 64))
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, x, edge_index, **kwargs):
        data = kwargs.get('data')
        data = data.detach().clone()
        x = F.elu(self.conv1(x, edge_index, data.edge_attr))
        weight = normalized_cut_2d(edge_index, data.pos)
        cluster = graclus(edge_index, weight, x.size(0))
        data.edge_attr = None
        data.x = x
        data.edge_index = edge_index
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def train(model, dataloader):
    model.train()

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data.x, data), data.y).backward()
        optimizer.step()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'current device: {device}')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2):
        train(model, train_loader)

    explainer = Explainer(
        model=model, algorithm=PGMExplainer(perturb_feature_list=[0],
                                            perturbation_mode="mean"),
        explanation_type='phenomenon', node_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph",
                          return_type="raw"))
    i = 0

    for explain_dataset in test_loader:
        explain_dataset.to(device)
        explanation = explainer(x=explain_dataset.x,
                                edge_index=explain_dataset.edge_index,
                                target=explain_dataset.y,
                                edge_attr=explain_dataset.edge_attr,
                                data=explain_dataset)
        for k in explanation.available_explanations:
            print(explanation[k])
        i += 1
        if i > 2:
            break
