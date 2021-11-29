import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index

loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5, node_idx=data.train_mask)
test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5, node_idx = data.test_mask)
val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5, node_idx = data.val_mask)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = F.nll_loss(out, data.y)


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_examples += 1
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    test_correct = 0
    test_total = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=-1)
        test_correct += pred.eq(data.y.to(device)).item()
        test_total += 1
    accs_test = test_correct / test_total
    
    val_correct = 0
    val_total = 0
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=-1)
        val_correct += pred.eq(data.y.to(device)).item()
        val_total += 1
    accs_val = val_correct / val_total
    
    
    
    return accs_test, accs_val


for epoch in range(1, 10):
    loss = train()
    test_acc, val_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f} Test: {test_acc:.4f}')
