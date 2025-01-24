import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0]

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Perform a 80/10/10 temporal link-level split:
perm = torch.argsort(data['user', 'movie'].time)
train_idx = perm[:int(0.8 * perm.size(0))]
val_idx = perm[int(0.8 * perm.size(0)):int(0.9 * perm.size(0))]
test_idx = perm[int(0.9 * perm.size(0)):]

edge_index = data['user', 'movie'].edge_index
kwargs = dict(
    data=data,
    num_neighbors=[20, 10],
    batch_size=1024,
    time_attr='time',
    temporal_strategy='last',
    num_workers=4,
    persistent_workers=True,
)
train_loader = LinkNeighborLoader(
    edge_label_index=(('user', 'movie'), edge_index[:, train_idx]),
    edge_label=data['user', 'movie'].edge_label[train_idx],
    edge_label_time=data['user', 'movie'].time[train_idx] - 1,
    shuffle=True,
    **kwargs,
)
val_loader = LinkNeighborLoader(
    edge_label_index=(('user', 'movie'), edge_index[:, val_idx]),
    edge_label=data['user', 'movie'].edge_label[val_idx],
    edge_label_time=data['user', 'movie'].time[val_idx] - 1,
    **kwargs,
)
test_loader = LinkNeighborLoader(
    edge_label_index=(('user', 'movie'), edge_index[:, test_idx]),
    edge_label=data['user', 'movie'].edge_label[test_idx],
    edge_label_time=data['user', 'movie'].time[test_idx] - 1,
    **kwargs,
)


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


def train():
    model.train()
    total_loss = total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'movie'].edge_label_index,
        )
        target = batch['user', 'movie'].edge_label.float()
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss * pred.size(0))
        total_examples += pred.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'movie'].edge_label_index,
        ).clamp(min=0, max=5)
        preds.append(pred)
        targets.append(batch['user', 'movie'].edge_label.float())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    rmse = (pred - target).pow(2).mean().sqrt()
    return float(rmse)


for epoch in range(1, 11):
    loss = train()
    val_rmse = test(val_loader)
    test_rmse = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val RMSE: {val_rmse:.4f}, '
          f'Test RMSE: {test_rmse:.4f}')
