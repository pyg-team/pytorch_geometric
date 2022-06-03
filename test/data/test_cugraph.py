import sys
sys.path += ['/work/pytorch_geometric', '/work/gaas/python']

from torch_geometric.data import Data
from torch_geometric.data.cugraph import CuGraphData
from torch_geometric.nn import GCNConv

import numpy as np

import torch
import torch.nn.functional as F

from gaas_client import GaasClient

client = GaasClient()
client.load_csv_as_edge_data('/work/cugraph/datasets/karate.csv', ['int32','int32','float32'], ['0','1'])

cd = CuGraphData(client)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)

data = cd.to(device)
data.x = torch.Tensor(np.ones((34,8))).type(torch.FloatTensor).to(device)
data.y = torch.Tensor(np.zeros(34)).type(torch.LongTensor).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == data.y).sum()
acc = int(correct) / 34
print(f'Accuracy: {acc:.4f}')