import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]

kwargs = {'batch_size': 1024, 'num_workers': 10, 'persistent_workers': True}

# Mimic ShadowGNN by setting the argument subgraph_type='induced'.
# This along with the num_neighbors argument will get the
# complete induced subgraph
train_loader = NeighborLoader(data, num_neighbors=[5] * 2,
                              input_nodes=data.train_mask,
                              subgraph_type='induced', **kwargs)
val_loader = NeighborLoader(data, num_neighbors=[5] * 2,
                            input_nodes=data.val_mask, subgraph_type='induced',
                            **kwargs)
test_loader = NeighborLoader(data, num_neighbors=[5] * 2,
                             input_nodes=data.test_mask,
                             subgraph_type='induced', **kwargs)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, root_n_id):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)

        # We merge both central node embeddings and subgraph embeddings:

        # x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)

        x = self.lin(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for data in tqdm(train_loader, desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.input_id)
        loss = F.cross_entropy(out[:data.batch_size], data.y[:data.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_examples += 1
    return total_loss / total_examples


@torch.no_grad()
def test(loader, desc='Eval'):
    model.eval()
    total_correct = total_examples = 0
    for data in tqdm(loader, desc=desc):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.input_id)
        total_correct += int(
            (out.argmax(dim=-1) == data.y).sum())
        total_examples += len(data.y)
    return total_correct / total_examples


for epoch in range(1, 51):
    loss = train()
    val_acc = test(val_loader, "Val")
    test_acc = test(test_loader, "Test")
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val accuracy: {val_acc:.4f} Test accuracy: {test_acc:.4f}')
