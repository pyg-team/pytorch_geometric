import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTEdgeSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_attr = 1. / degree(row, data.num_nodes)[row]

loader = GraphSAINTEdgeSampler(data, batch_size=6000, num_steps=200,
                               sample_coverage=1000,
                               save_dir=dataset.processed_dir, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = SAGEConv(in_channels, hidden_channels, concat=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, concat=True)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, concat=True)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    model.set_aggr('add')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        loss = F.nll_loss(out, data.y, reduction='none')
        loss = (loss * data.node_norm).sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


def train_full():
    model.eval()
    model.set_aggr('mean')

    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y.to(device)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


for epoch in range(1, 31):
    loss = train()
    accs = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
          f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
