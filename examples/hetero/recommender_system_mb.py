import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from tqdm import tqdm
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn.pool import MIPSKNNIndex
from ordered_set import OrderedSet

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import SAGEConv

parser = argparse.ArgumentParser()
parser.add_argument('--visualize_emb', action='store_true',
                    help='Epoch-wise visualization of embeddings')
parser.set_defaults(visualize_emb=False)
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
        self.user_to_item = Linear(hidden_channels, hidden_channels)
        self.item_to_user = Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        u_to_i = self.user_to_item(z_dict['user'][row])
        i_to_u = self.user_to_item(z_dict['movie'][row])
                                   
        z = torch.cat([u_to_i, i_to_u], dim=-1)
        # z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

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
    if args.visualize_emb:
        tsne = TSNE(random_state=1, n_iter=1000, early_exaggeration=20)
        embs = tsne.fit_transform(model.encoder(val_data.x_dict, 
            val_data.edge_index_dict)['movie'].detach().numpy())
        plt.scatter(embs[:,0], embs[:,1], alpha=0.1)
        plt.savefig('./fig-' + str(epoch) + '.png')
        plt.clf()
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

k=20
embs = model.encoder(val_data.x_dict, 
    val_data.edge_index_dict)
mipsknn = MIPSKNNIndex(embs['movie'])
embs_user_to_movie = model.decoder.user_to_item(embs['user'])
knn_score, knn_index = mipsknn.search(embs_user_to_movie, embs_user_to_movie.size(0))

def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items
user_pos_items = get_user_positive_items(train_data['user', 'movie'].edge_index)

real_recs = {}
for user in range(val_data['user'].x.size(0)):
    real_recs[user] = list(OrderedSet(knn_index[user].tolist()) - OrderedSet(user_pos_items[user]))[:k]


