import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import pmlp
from torch_geometric.utils import subgraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'Cora'
transform = T.NormalizeFeatures()
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]

model = pmlp.PMLP(in_channels=data.x.shape[1], hidden_channels=256)

criterion = nn.NLLLoss()

data = data.to(device)
model = model.to(device)

runs = 10
epochs = 100

best_acc = float('-inf')
train_times = []
for t in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3,
                                 lr=0.01)

    for epoch in range(epochs):
        train_start = time.time()

        # training
        model.train()
        optimizer.zero_grad()
        train_idx = torch.as_tensor(torch.where(data.train_mask)[0]).to(device)
        out = model(data.x, dataset.edge_index.contiguous())
        out = F.log_softmax(out, dim=1)
        true_label = F.one_hot(data.y,
                               data.y.max() + 1).squeeze(1).to(torch.float)

        loss = nn.KLDivLoss(reduction='batchmean')(out[train_idx],
                                                   true_label[train_idx])
        loss.backward()
        optimizer.step()

        train_time = time.time() - train_start
        train_times.append(train_time)

        # testing
        model.eval()

        valid_idx = torch.as_tensor(torch.where(data.val_mask)[0]).to(device)
        valid_idx = torch.cat([train_idx, valid_idx], dim=-1)
        valid_edge_idx = subgraph(valid_idx, dataset.edge_index)[0]
        out = model(data.x[valid_idx], valid_edge_idx.contiguous())
        out = F.log_softmax(out, dim=1)
        y_true = data.y[valid_idx].detach().cpu().numpy()
        y_pred = out.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
        acc_val = len(
            np.where(y_true == y_pred.T.tolist()[0])[0]) / len(y_true)

        test_idx = torch.as_tensor(torch.where(data.test_mask)[0]).to(device)
        test_idx = torch.cat([train_idx, valid_idx], dim=-1)
        test_edge_idx = subgraph(test_idx, dataset.edge_index)[0]
        out = model(data.x[test_idx], test_edge_idx.contiguous())
        out = F.log_softmax(out, dim=1)
        y_true = data.y[test_idx].detach().cpu().numpy()
        y_pred = out.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
        acc_test = len(
            np.where(y_true == y_pred.T.tolist()[0])[0]) / len(y_true)

        if acc_test > best_acc:
            best_acc = acc_test

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, '
                  f'Loss: {loss: .3f}, '
                  f'Val: {acc_val: .3f}, '
                  f'Test: {acc_test: .3f}')

train_time = sum(train_times) / len(train_times)
print(f'Best Acc: {best_acc: .3f}, Train time: {train_time: .6f}')
