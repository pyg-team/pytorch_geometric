import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]
loader = NeighborSampler(
    data,
    size=[25, 10],
    num_hops=2,
    batch_size=1000,
    shuffle=True,
    add_self_loops=True)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=False)
        self.conv2 = SAGEConv(16, out_channels, normalize=False)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1(x, data.edge_index, data.size))
        x = F.dropout(x, training=self.training)
        data = data_flow[1]
        x = self.conv2(x, data.edge_index, data.size)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def test(mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


for epoch in range(1, 201):
    loss = train()
    test_acc = test(data.test_mask)
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
