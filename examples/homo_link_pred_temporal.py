import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric as torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import SAGEConv

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/MovieLens')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

#Add timestamp to the movie and users to be zeros.
# data['movie']['node_time'] = torch.zeros(len(data['movie']['x']), dtype=torch.long)
# data['user']['node_time'] = torch.zeros(len(data['user']['x']), dtype=torch.long)
# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)

train_data = train_data.to_homogeneous()
val_data = val_data.to_homogeneous()
test_data = test_data.to_homogeneous()

train_dataloader = LinkNeighborLoader(data=train_data, num_neighbors=[5, 5, 5],
                                      neg_sampling_ratio=1,
                                      edge_label_index=train_data.edge_index,
                                      edge_label_time=train_data.timestamp,
                                      batch_size=2048, shuffle=True,
                                      time_attr='timestamp')
val_dataloader = LinkNeighborLoader(data=val_data, num_neighbors=[5, 5, 5],
                                    neg_sampling_ratio=1,
                                    edge_label_index=val_data.edge_index,
                                    edge_label_time=val_data.timestamp,
                                    batch_size=4096, shuffle=True,
                                    time_attr='timestamp')
test_dataloader = LinkNeighborLoader(data=test_data, num_neighbors=[5, 5, 5],
                                     neg_sampling_ratio=1,
                                     edge_label_index=test_data.edge_index,
                                     edge_label_time=test_data.timestamp,
                                     batch_size=4096, shuffle=True,
                                     time_attr='timestamp')
# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if args.use_weighted_loss:
    weight = torch.bincount(train_data.edge_label)
    weight = weight.max() / weight
else:
    weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z[row], z[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_label_index)


model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# model = torch_geometric.compile(model)

import time

from tqdm import tqdm


def train(train_dl, val_dl):
    model.train()
    pbar = tqdm(train_dl, desc='Train')
    t0 = time.time()
    train_time = 0
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_label_index)
        target = batch.edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        t1 = time.time()
        train_time += t1 - t0
        t0 = time.time()
    print(f'train time = {train_time}')
    return float(loss)


@torch.no_grad()
def test(dl, desc='val'):
    model.eval()
    rmse = 0
    t0 = time.time()
    val_time = 0
    for batch in tqdm(dl, desc=desc):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = batch.edge_label.float()
        rmse += F.mse_loss(pred, target).sqrt()
        t1 = time.time()
        val_time += t1 - t0
        t0 = time.time()
    print(f'{desc} time = {val_time}')
    return float(rmse)


for epoch in range(1, 31):
    loss = train(train_dl=train_dataloader, val_dl=val_dataloader)
    # train_rmse = test(val_dataloader)
    val_rmse = test(val_dataloader, 'val')
    # test_rmse = test(test_dataloader, 'test')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},'
          f'Val: {val_rmse:.4f}')
    #   f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
