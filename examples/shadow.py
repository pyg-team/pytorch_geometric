import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.nn import SAGEConv, global_mean_pool

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
train_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                 node_idx=data.train_mask, **kwargs)
val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                               node_idx=data.val_mask, **kwargs)
test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                                node_idx=data.test_mask, **kwargs)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, root_n_id):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)

        # We merge both central node embeddings and subgraph embeddings:
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)

        x = self.lin(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_examples += data.num_graphs
    return total_correct / total_examples


for epoch in range(1, 51):
    loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val: {val_acc:.4f} Test: {test_acc:.4f}')
