import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear, Embedding
from tqdm import tqdm
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn.pool import MIPSKNNIndex
from ordered_set import OrderedSet
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import json
import random

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.metrics import LinkPredPrecision

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('-k', '--k', type=int, default=20,
                    help='number of predictions per user to make')
parser.add_argument('--visualize_emb', action='store_true',
                    help='Epoch-wise visualization of embeddings')
parser.set_defaults(visualize_emb=False)
parser.add_argument('--profile_code', action='store_true',
                    help='Display timing information')
parser.set_defaults(profile_code=False)
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

print(args)
random.seed(42)
torch.manual_seed(42)

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

# Add node ids
data['movie'].n_id = torch.arange(0, data['movie'].x.size(0))
data['user'].n_id = torch.arange(0, data['user'].x.size(0))

# Use only edges that have high ratings (>=4)
rating_threshold = 4
edge_names = [('user', 'rates', 'movie'), ('movie', 'rev_rates', 'user')]
filter_indices = torch.where(data[edge_names[0]].edge_label >=
                             rating_threshold)[0]
for edge_name in edge_names:
    for attr in data[edge_name]:
        if len(data[edge_name][attr].size()) == 1:
            data[edge_name][attr] = data[edge_name][attr][filter_indices]
        else:
            data[edge_name][attr] = data[edge_name][attr][:, filter_indices]

# Perform a temporal link-level split into training, validation, and test edges:
split_ratio = [0.8, 0.1, 0.1]
assert sum(split_ratio) == 1
_, perm = torch.sort(data[('user', 'rates', 'movie')].time)
for edge_name in edge_names:
    for attr in data[edge_name]:
        if len(data[edge_name][attr].size()) == 1:
            data[edge_name][attr] = data[edge_name][attr][perm]
        else:
            data[edge_name][attr] = data[edge_name][attr][:, perm]
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


BATCH_SIZE = 512

train_dataloader = LinkNeighborLoader(
    data=train_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      train_data[('user', 'rates',
                                  'movie')].edge_index),
    edge_label_time=train_data[('user', 'rates', 'movie')].time,
    batch_size=BATCH_SIZE, shuffle=True, time_attr='time')

val_dataloader = LinkNeighborLoader(
    data=val_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      val_data[('user', 'rates',
                                'movie')].edge_index),
    edge_label_time=val_data[('user', 'rates', 'movie')].time,
    batch_size=BATCH_SIZE, shuffle=True, time_attr='time')

test_dataloader = LinkNeighborLoader(
    data=test_data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      test_data[('user', 'rates',
                                 'movie')].edge_index),
    edge_label_time=test_data[('user', 'rates', 'movie')].time,
    batch_size=BATCH_SIZE, shuffle=True, time_attr='time')

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
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
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
        i_to_u = self.item_to_user(z_dict['movie'][col])

        z = torch.cat([u_to_i, i_to_u], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)
        # return z


class Model(torch.nn.Module):
    def __init__(self, num_users, num_movies, hidden_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels, device=device)
        self.movie_emb = Embedding(num_movies, hidden_channels, device=device)
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.user_emb.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.movie_emb.weight, gain=1)

    def forward(self, user_id, movie_id, x_dict, edge_index_dict,
                edge_label_index):
        x_dict['user'] = torch.cat(
            [self.user_emb(user_id), x_dict['user']], dim=-1)
        x_dict['movie'] = torch.cat(
            [self.movie_emb(movie_id), x_dict['movie']], dim=-1)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(
    num_users=data['user'].x.size(0),
    num_movies=data['movie'].x.size(0),
    hidden_channels=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)


def get_user_positive_items(edge_index):
    # Get all the movies that the user has actually watched
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items
    
def get_embeddings(model, val_data):
    # Get user and movie embeddings from the model
    x_dict = {}
    x_dict['user'] = torch.cat([model.user_emb(val_data['user'].n_id),
                                val_data.x_dict['user']], dim=-1)
    x_dict['movie'] = torch.cat([model.movie_emb(val_data['movie'].n_id),
                                 val_data.x_dict['movie']], dim=-1)
    embs = model.encoder(x_dict,
                         val_data.edge_index_dict)
    movie_embs = embs['movie']
    embs_user_to_movie = embs['user']
    embs_user_to_movie = model.decoder.user_to_item(embs['user'])
    movie_embs = model.decoder.item_to_user(embs['movie'])
    return movie_embs, embs_user_to_movie

def make_recommendations(model,
                         train_data,
                         val_data,
                         k,
                         write_to_file=False):

    movie_embs, user_embs = get_embeddings(model, val_data)
    mipsknn = MIPSKNNIndex(movie_embs)
    knn_score, knn_index = mipsknn.search(
        user_embs, movie_embs.size(0))

    # Obtain the movies that each user has already
    # watched. These movies need to be removed from
    # recommendations
    user_pos_items = get_user_positive_items(
        train_data['user', 'movie'].edge_index)

    # Get the recommendations for user; only those
    # that the user is yet to watch
    real_recs = {}
    for user in range(val_data['user'].x.size(0)):
        try:
            real_recs[user] = list(
                OrderedSet(knn_index[user].tolist()) -
                OrderedSet(user_pos_items[user]))[
                : k]
        except KeyError:
            real_recs[user] = knn_index[user].tolist()[:k]

    # Get the ground truth
    val_user_pos_items = get_user_positive_items(
        val_data['user', 'movie'].edge_index)

    # Let's write out the recommendations into a file
    if write_to_file:
        movie_path = osp.join(path, './raw/ml-latest-small/movies.csv')
        rating_path = osp.join(path, './raw/ml-latest-small/ratings.csv')
        # load user and movie nodes

        def load_node_csv(path, index_col):
            df = pd.read_csv(path, index_col=index_col)
            return df

        movie_id_to_name = load_node_csv(movie_path, index_col='movieId')
        with open('recommendations.txt', 'w') as f:
            for user in range(val_data['user'].x.size(0)):
                try:
                    if user in val_user_pos_items:
                        f.write('\nPred\n')
                        f.write(json.dumps(
                            movie_id_to_name.iloc[real_recs[user]].to_dict()))
                        f.write('\nGT\n')
                        f.write(
                            json.dumps(
                                movie_id_to_name.iloc
                                [val_user_pos_items[user]].to_dict()))
                except KeyError:
                    pass
    # Obtain some metrics
    hits = 0
    user_hits = 0
    for user in val_user_pos_items:
        correct_preds = len(set(real_recs[user]).intersection(
            set(val_user_pos_items[user])))
        hits += correct_preds
        user_hits += correct_preds > 0
    
    print(f"Total hits at top-{k} = {hits}")
    print(
        f"Total user hits at top-{k} = {user_hits}/{len(val_user_pos_items)}")
    
    return real_recs


def visualize(model, val_data, epoch):
    tsne = TSNE(random_state=1, n_iter=1000, early_exaggeration=20)
    movie_embs, user_embs = get_embeddings(model, val_data)
    emb_reduced = tsne.fit_transform(torch.vstack((movie_embs,
                                                   user_embs,
                                                   )).detach().numpy())
    plt.scatter(emb_reduced[:movie_embs.size(0), 0],
                emb_reduced[:movie_embs.size(0), 1],
                alpha=0.1)
    plt.scatter(emb_reduced[movie_embs.size(0):, 0],
                emb_reduced[movie_embs.size(0):, 1],
                alpha=0.1,
                marker='x')
    plt.savefig('./fig-' + str(epoch) + '.png')
    plt.clf()


def randomize_inputs(batch):
    perm = torch.randperm(len(batch['user', 'movie'].edge_label))
    batch[edge_names[0]]['edge_label'] = batch[edge_names[0]]['edge_label'][perm]
    batch[edge_names[0]]['edge_label_index'] = batch[edge_names[0]][
        'edge_label_index'][:, perm]
    return batch


def train(train_dl, val_dl, val_data, epoch):
    model.train()
    pbar = tqdm(train_dl, desc='Train')
    t0 = time.time()
    train_time = 0

    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        batch = randomize_inputs(batch)
        pred = model(batch['user'].n_id,
                     batch['movie'].n_id,
                     batch.x_dict, batch.edge_index_dict,
                     batch['user', 'movie'].edge_label_index)
        target = batch['user', 'movie'].edge_label
        loss = weighted_mse_loss(pred, target, weight)
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #                                           pred.view(-1), target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        t1 = time.time()
        train_time += t1 - t0
        t0 = time.time()

    if args.profile_code is True:
        print(f'train time = {train_time}')

    if args.visualize_emb:
        visualize(model, val_data, epoch)

    return float(loss)


@torch.no_grad()
def test(dl, desc='val'):
    model.eval()
    rmse = 0
    t0 = time.time()
    val_time = 0
    preds, targets = [], []
    for batch in tqdm(dl, desc=desc):
        optimizer.zero_grad()
        batch = randomize_inputs(batch)
        pred = model(batch['user'].n_id, batch['movie'].n_id, batch.x_dict,
                     batch.edge_index_dict,
                     batch['user', 'movie'].edge_label_index)  
                     # .sigmoid().view(-1).cpu()
        pred = pred.clamp(min=0, max=5)
        target = batch['user', 'movie'].edge_label.float()
        preds.append(pred)
        targets.append(target)

        rmse += F.mse_loss(pred, target).sqrt()
        t1 = time.time()
        val_time += t1 - t0
        t0 = time.time()
    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    acc = accuracy_score(target, pred > 0.5)
    roc = roc_auc_score(target, pred)
    if args.profile_code is True:
        print(f'{desc} time = {val_time}')
    return float(rmse), roc, acc

def compute_metrics(real_recs, val_dataloader, val_data, k, desc):
    metric = LinkPredPrecision(k)
    real_recs_list = []
    [real_recs_list.append(rec) for id, rec in real_recs.items()]
    rec_tensor = torch.tensor(real_recs_list)

    for batch in val_dataloader:
        node_id = batch['user'].n_id
        edge_label_index = batch['user', 'movie'].edge_label_index
        mask = torch.isin(edge_label_index[0], node_id)
        y_batch, y_index = edge_label_index[:, mask]
        arange = torch.empty(batch['user'].x.size(0), dtype=node_id.dtype)
        arange[node_id] = torch.arange(node_id.numel())
        y_batch = arange[y_batch]
        metric.update(rec_tensor[node_id], (y_batch, y_index))
    return metric.compute()

EPOCHS = args.epochs
for epoch in range(0, EPOCHS):
    loss = train(train_dl=train_dataloader, val_dl=val_dataloader,
                 val_data=val_data, epoch=epoch)
    val_rmse = test(val_dataloader, 'val')
    real_recs = make_recommendations(model, train_data, val_data, args.k, False)
    val_precision_at_k = compute_metrics(real_recs, val_dataloader, val_data, args.k, 'val prec@k')
    print(
        f'Epoch: {epoch:03d}, Loss: {loss:.4f},'
        f'Val RMSE: {val_rmse[0]:.4f}, Val ROC: {val_rmse[1]:.4f} Val Acc: {val_rmse[2]:.4f}')
    print(f'Val precision@{args.k} = {val_precision_at_k:.3E}')

# Get results on test split
test_rmse = test(test_dataloader, 'test')
print(f'test RMSE: {test_rmse[0]:.4f}, test ROC: {test_rmse[1]:.4f} test Acc: {test_rmse[2]:.4f}')
real_recs = make_recommendations(model, train_data, test_data, args.k, False)
test_precision_at_k = compute_metrics(real_recs, test_dataloader, test_data, args.k, 'test prec@k')
print(f'test precision@{args.k} = {test_precision_at_k:.3E}')

# Save the model for good measure
torch.save(model.state_dict(), "./model.bin")
