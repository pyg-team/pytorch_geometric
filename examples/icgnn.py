import os.path as osp

import torch
from torch.nn import CrossEntropyLoss

from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import ICGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
dataset = Planetoid(root=path, name='cora')
data = dataset[0].to(device)
loss_fn = CrossEntropyLoss()
num_nodes = data.x.shape[0]
num_features = data.x.shape[1]
num_classes = data.y.max() + 1

model = ICGNN(
    num_nodes=num_nodes,
    K=10,
    feature_scale=1,
    in_channel=num_features,
    hidden_channel=num_features,
    out_channel=num_classes,
    num_layers=2,
    dropout=0.1,
    activation='relu',
).to(device)
icg_optimizer = torch.optim.Adam(model.icg_parameters(), lr=0.1)
nn_optimizer = torch.optim.Adam(model.nn_parameters(), lr=0.01)


def icg_fit() -> float:
    icg_optimizer.zero_grad()
    loss = model.icg_fit_loss(x=data.x, edge_index=data.edge_index)
    loss.backward()
    icg_optimizer.step()
    return loss.item()


def train() -> float:
    model.train()
    nn_optimizer.zero_grad()
    scores = model(data.x)
    loss = loss_fn(scores, data.y)
    loss.backward()
    nn_optimizer.step()
    return loss.item()


@torch.no_grad()
def test() -> float:
    model.eval()
    scores = model(data.x)
    predictions = scores.argmax(dim=1)
    return (predictions == data.y).float().mean().item()


print('ICG fitting')
for epoch in range(1, 301):
    loss = icg_fit()
    print(f'Epoch: {epoch: 03d}, ICG approximation loss: {loss: .4f}')

print('\nNN training')
for epoch in range(1, 11):
    loss = train()
    accuracy = test()
    print(f'Epoch: {epoch: 03d}, Loss: {loss: .4f}, accuracy: {accuracy}')
