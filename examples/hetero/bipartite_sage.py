import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0]
data['user'].x = torch.arange(data['user'].num_nodes)
data['user', 'movie'].edge_label = data['user', 'movie'].edge_label.float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)

# Generate the co-occurence matrix of movies<>movies:
metapath = [('movie', 'rev_rates', 'user'), ('user', 'rates', 'movie')]
train_data = T.AddMetaPaths(metapaths=[metapath])(train_data)

# Apply normalization to filter the metapath:
_, edge_weight = gcn_norm(
    train_data['movie', 'movie'].edge_index,
    num_nodes=train_data['movie'].num_nodes,
    add_self_loops=False,
)
edge_index = train_data['movie', 'movie'].edge_index[:, edge_weight > 0.002]

train_data['movie', 'metapath_0', 'movie'].edge_index = edge_index
val_data['movie', 'metapath_0', 'movie'].edge_index = edge_index
test_data['movie', 'metapath_0', 'movie'].edge_index = edge_index


class MovieGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        movie_x = self.conv1(
            x_dict['movie'],
            edge_index_dict[('movie', 'metapath_0', 'movie')],
        ).relu()

        user_x = self.conv2(
            (x_dict['movie'], x_dict['user']),
            edge_index_dict[('movie', 'rev_rates', 'user')],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict[('movie', 'rev_rates', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.movie_encoder = MovieGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)
        z_dict['movie'] = self.movie_encoder(
            x_dict['movie'],
            edge_index_dict[('movie', 'metapath_0', 'movie')],
        )
        return self.decoder(z_dict['user'], z_dict['movie'], edge_label_index)


model = Model(data['user'].num_nodes, hidden_channels=64, out_channels=64)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data['user', 'movie'].edge_label_index,
    )
    loss = F.mse_loss(out, train_data['user', 'movie'].edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    out = model(
        data.x_dict,
        data.edge_index_dict,
        data['user', 'movie'].edge_label_index,
    ).clamp(min=0, max=5)
    rmse = F.mse_loss(out, data['user', 'movie'].edge_label).sqrt()
    return float(rmse)


for epoch in range(1, 701):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
