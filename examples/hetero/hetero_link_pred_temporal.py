import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from tqdm import tqdm
import copy

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import SAGEConv

parser = argparse.ArgumentParser()
parser.add_argument('--profile_code', action='store_true',
                    help='Display timing information')
parser.set_defaults(profile_code=False)
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Perform a temporal link-level split into training, validation, and test edges:
split_ratio = [0.8, 0.1, 0.1]

_, perm = torch.sort(data[('user', 'rates', 'movie')].time)
edge_names = [('user', 'rates', 'movie'), ('movie', 'rev_rates', 'user')]
for edge_name in edge_names:
    for attr in data[edge_name]:
        if len(data[edge_name][attr].size()) == 1:
            data[edge_name][attr] = data[edge_name][attr][perm]
        else:
            data[edge_name][attr] = data[edge_name][attr][:,perm]
num_edges = len(data[edge_names[0]].time)

splits = [int(i*num_edges) for i in split_ratio]
splits[2] += num_edges-sum(splits)
train_data = copy.deepcopy(data)
val_data = copy.deepcopy(data)
test_data = copy.deepcopy(data)
graphs = [train_data, val_data, test_data]
for edge_name in edge_names:
    for attr in data[edge_name]:
        dim = len(data[edge_name][attr].size()) - 1
        data_splits = torch.split(data[edge_name][attr], splits, dim)
        for i, graph in enumerate(graphs):
            graph[edge_name][attr] = data_splits[i]


train_dataloader = LinkNeighborLoader(
    data=train_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      train_data[('user', 'rates',
                            'movie')].edge_index),
    edge_label_time=train_data[('user', 'rates', 'movie')].time,
    batch_size=4096, shuffle=True, time_attr='time')
val_dataloader = LinkNeighborLoader(
    data=val_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      val_data[('user', 'rates',
                            'movie')].edge_index),
    edge_label_time=val_data[('user', 'rates', 'movie')].time,
    batch_size=4096, shuffle=True, time_attr='time')
test_dataloader = LinkNeighborLoader(
    data=test_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      test_data[('user', 'rates',
                            'movie')].edge_index),
    edge_label_time=test_data[('user', 'rates', 'movie')].time,
    batch_size=4096, shuffle=True, time_attr='time')
# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if args.use_weighted_loss:
    weight = torch.bincount(data['user', 'movie'].edge_label)
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

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(train_dl, val_dl, val_data, epoch):
    model.train()
    pbar = tqdm(train_dl, desc='Train')
    t0 = time.time()
    train_time = 0
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict,
                     batch['user', 'movie'].edge_label_index)
        target = batch['user', 'movie'].edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        t1 = time.time()
        train_time += t1 - t0
        t0 = time.time()
    if args.profile_code is True:
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
        pred = model(batch.x_dict, batch.edge_index_dict,
                     batch['user', 'movie'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = batch['user', 'movie'].edge_label.float()
        rmse += F.mse_loss(pred, target).sqrt()
        t1 = time.time()
        val_time += t1 - t0
        t0 = time.time()
    if args.profile_code is True:
        print(f'{desc} time = {val_time}')
    return float(rmse)

EPOCHS = 10
for epoch in range(0, EPOCHS):
    loss = train(train_dl=train_dataloader, val_dl=val_dataloader, val_data=val_data, epoch=epoch)
    # train_rmse = test(val_dataloader)
    val_rmse = test(val_dataloader, 'val')
    test_rmse = test(test_dataloader, 'test')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},'
          f'Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
