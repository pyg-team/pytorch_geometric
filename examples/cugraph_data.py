import sys
sys.path += ['/work/pytorch_geometric', '/work/gaas/python']
# assumes cugraph package is installed

from gaas_client import GaasClient
from torch_geometric.data import Data
from torch_geometric.data.cugraph import CuGraphData
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


client = GaasClient()
client.load_csv_as_vertex_data('/tmp/snap/email-Eu-core-department-labels.txt', dtypes=['int32','int32'],vertex_col_name='id', names=['id','dept'])
client.load_csv_as_edge_data('/tmp/snap/email-Eu-core.txt', ['int32','int32'], ['0','1'])
cd = CuGraphData(client)

N = 42 # The number of classes
C = N*4

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(cd.num_node_features - 1, C)
        self.conv2 = GCNConv(C, int(C/2))
        self.conv3 = GCNConv(int(C/2), N)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)

data = cd.to(device)
data.x = data.id[-1] # for now have to convert to local torch Tensor
data.y = data.dept[-1][:,0].to(torch.long) # for now have to convert to local torch Tensor

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
acc = int(correct) / cd.num_nodes
print(f'Accuracy: {acc:.4f}')
