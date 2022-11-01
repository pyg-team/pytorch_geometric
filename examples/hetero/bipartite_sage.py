import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
user_sim_dict_path = 'swing_user_sim_dict.json'
dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
data = dataset[0]
data['user'].x = torch.arange(data['user'].num_nodes)
data = data.to(device)
del data['user'].num_nodes

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


def to_u2i_mat(edge_index, i_num, u_num):
    # Convert bipartite edge_index format to matrix format
    u2imat = to_dense_adj(edge_index)[0]

    return u2imat[:u_num, :i_num]


def get_coocur_mat(train_mat, num_neighbors):
    # Generate the co-occurrence matrix with normalisation and top-k filtering
    A = train_mat.T @ train_mat
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))

    items_D = torch.sum(A, axis=0).reshape(-1)
    users_D = torch.sum(A, axis=1).reshape(-1)
    beta_uD = (1 / torch.sqrt(users_D + 1)).reshape(-1, 1)
    beta_iD = (1 / torch.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = beta_uD @ beta_iD

    norm_A = all_ii_constraint_mat * A
    res_sim_mat, res_mat = torch.topk(norm_A, num_neighbors, dim=1)

    return res_mat.long(), res_sim_mat.float()


def get_i2i_sim_graph(i2imat, i2isimmat):
    # Generate the i2i graph from the co-occurrence matrix with filters
    mask, len = i2isimmat > 1e-5, i2imat[0].size(0)
    mask = torch.tensor([
        torch.zeros(len, dtype=torch.bool).tolist()
        if v[0] != i else mask[i].tolist() for i, v in enumerate(i2imat)
    ])
    i2imat[~mask] = -1
    i2i_edge_idx = [[i, j] for i, v in enumerate(i2imat[:, 1:]) for j in v
                    if j >= 0]

    return torch.tensor(i2i_edge_idx).T


u2i_mat = to_u2i_mat(train_data.edge_index_dict[('user', 'rates', 'movie')],
                     train_data['movie'].x.shape[0],
                     train_data['user'].x.shape[0])
i2i_mat, i2i_sim_mat = get_coocur_mat(u2i_mat, 9)
i2i = get_i2i_sim_graph(i2i_mat, i2i_sim_mat)

# Add the generated i2i graph for high-order information
train_data[('movie', 'sims', 'movie')].edge_index = i2i.to(device)
val_data[('movie', 'sims', 'movie')].edge_index = i2i.to(device)
test_data[('movie', 'sims', 'movie')].edge_index = i2i.to(device)

# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if args.use_weighted_loss:
    weight = torch.bincount(train_data['user', 'movie'].edge_label)
    weight = weight.max() / weight
else:
    weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(-1, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, h, edge_index):
        h = self.conv1(h, edge_index).relu()
        h = self.lin1(h).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.lin2(h)
        return h


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        h = x_dict['movie']
        h = self.conv1(h, edge_index_dict[('movie', 'sims', 'movie')]).relu()
        h = self.lin1(h).relu()

        x_dict['user'] = self.conv2([x_dict['movie'], x_dict['user']],
                                    edge_index_dict[('movie', 'rev_rates',
                                                     'user')]).relu()
        x_dict['user'] = self.lin2(x_dict['user']).relu()

        x_dict['user'] = self.conv3([h, x_dict['user']],
                                    edge_index_dict[('movie', 'rev_rates',
                                                     'user')]).relu()
        x_dict['user'] = self.lin3(x_dict['user'])
        return x_dict['user']


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
    def __init__(self, input_size, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(input_size, hidden_channels, device=device)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        z_dict['movie'] = self.item_encoder(
            x_dict['movie'], edge_index_dict[('movie', 'sims', 'movie')])
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict, edge_label_index)


model = Model(input_size=data['user'].x.size(0), hidden_channels=64,
              out_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


for epoch in range(1, 701):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
