import os.path as osp

import torch
from torch.nn import CrossEntropyLoss, ModuleList

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.models import CoGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
dataset = Planetoid(root=path, name='cora')
data = dataset[0].to(device)
loss_fn = CrossEntropyLoss()
num_features = data.x.shape[1]
num_classes = data.y.max() + 1

# Use all message passing edges as training labels:
env_layer1 = GCNConv(
    in_channels=num_features,
    out_channels=num_features,
)

env_layer2 = GCNConv(
    in_channels=num_features,
    out_channels=num_classes,
)

action_net = GCN(
    in_channels=num_features,
    out_channels=4,
    hidden_channels=num_features,
    num_layers=2,
)

model = CoGNN(
    env_net=ModuleList([env_layer1, env_layer2]),
    action_net=action_net,
    env_activation='relu',
    temp=0.1,
    dropout=0.1,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    scores = model(data.x, data.edge_index)
    loss = loss_fn(scores, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test() -> float:
    model.eval()
    scores = model(data.x, data.edge_index)
    predictions = scores.argmax(dim=1)
    return (predictions == data.y).float().mean().item()


for epoch in range(1, 101):
    loss = train()
    accuracy = test()
    print(f'Epoch: {epoch: 03d}, Loss: {loss: .4f}, accuracy: {accuracy}')
