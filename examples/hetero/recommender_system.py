import argparse
import copy
import json
import os.path as osp
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.nn import Embedding, Linear
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.metrics import LinkPredNDCG, LinkPredPrecision
from torch_geometric.nn.pool import MIPSKNNIndex

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('-k', '--k', type=int, default=20,
                    help='number of predictions per user to make')
parser.add_argument('--visualize_emb', action='store_true',
                    help='Epoch-wise visualization of embeddings')
parser.set_defaults(visualize_emb=False)
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

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Add node ids
data['movie'].n_id = torch.arange(0, data['movie'].num_nodes)
data['user'].n_id = torch.arange(0, data['user'].num_nodes)

# Use only edges that have high ratings (>=4)
rating_threshold = 4
edge_names = [('user', 'rates', 'movie'), ('movie', 'rev_rates', 'user')]
filter_indices = torch.where(
    data[edge_names[0]].edge_label >= rating_threshold)[0]
for edge_name in edge_names:
    for attr in data[edge_name]:
        if len(data[edge_name][attr].size()) == 1:
            data[edge_name][attr] = data[edge_name][attr][filter_indices]
        else:
            data[edge_name][attr] = data[edge_name][attr][:, filter_indices]

# Perform a temporal link-level split into train, val, and test edges:
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

splits = [int(i * num_edges) for i in split_ratio]
splits[2] += num_edges - sum(splits)
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
limiting_time = min(val_data['user','movie'].time)-1
train_dataloader = LinkNeighborLoader(
    data=data, num_neighbors=[5, 5, 5], neg_sampling_ratio=1,
    edge_label_index=(('user', 'rates', 'movie'),
                      train_data[('user', 'rates', 'movie')].edge_index),
    edge_label_time= torch.ones(data['user', 'movie'].time.size(), dtype=torch.long)*limiting_time,
    batch_size=BATCH_SIZE, shuffle=True, time_attr='time')


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)
        self.to_hidden = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = self.to_hidden(x)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        u = z_dict['user'][row]
        i = z_dict['movie'][col]

        z = (u * i).sum(dim=1)

        return z.view(-1)


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
        x_dict['user'] = self.user_emb(user_id)
        x_dict['movie'] = torch.cat(
            [self.movie_emb(movie_id), x_dict['movie']], dim=-1)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(num_users=data['user'].num_nodes,
              num_movies=data['movie'].num_nodes,
              hidden_channels=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


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


def write_recs(real_recs, val_data):
    movie_path = osp.join(path, './raw/ml-latest-small/movies.csv')

    # load user and movie nodes

    def load_node_csv(path, index_col):
        df = pd.read_csv(path, index_col=index_col)
        return df

    movie_id_to_name = load_node_csv(movie_path, index_col='movieId')

    # Get the ground truth
    val_user_pos_items = get_user_positive_items(val_data['user',
                                                          'movie'].edge_index)
    with open('recommendations.txt', 'w') as f:
        for user in range(val_data['user'].num_nodes):
            try:
                if user in val_user_pos_items:
                    f.write('\nPred\n')
                    f.write(
                        json.dumps(
                            movie_id_to_name.iloc[real_recs[user]].to_dict()))
                    f.write('\nGT\n')
                    f.write(
                        json.dumps(movie_id_to_name.iloc[
                            val_user_pos_items[user]].to_dict()))
            except KeyError:
                pass


def get_embeddings(model, val_data):
    # Get user and movie embeddings from the model
    x_dict = {}
    x_dict['user'] = model.user_emb(val_data['user'].n_id)
    x_dict['movie'] = torch.cat(
        [model.movie_emb(val_data['movie'].n_id), val_data.x_dict['movie']],
        dim=-1)
    embs = model.encoder(x_dict, val_data.edge_index_dict)
    return embs['movie'], embs['user']


def make_recommendations(model, inference_dataloader, k):
    real_recs = {}
    for batch in inference_dataloader:
        movie_embs, user_embs = get_embeddings(model, batch)
        mipsknn = MIPSKNNIndex(movie_embs)
        knn_score, knn_index_ = mipsknn.search(
            user_embs, k, batch['user', 'movie'].edge_index)
        knn_index_translated = batch['movie'].n_id[knn_index_[:][:]]

        # Get the recommendations for user;
        for i, user in enumerate(batch['user'].n_id.detach().numpy()):
            rec = knn_index_translated[i, :].tolist()
            real_recs[user] = rec[:k]

    return real_recs


def visualize(model, train_data, val_data, real_recs, epoch, users_list=[338]):
    tsne = TSNE(random_state=1, n_iter=1000, early_exaggeration=20)
    movie_embs, user_embs = get_embeddings(model, val_data)
    emb_reduced = tsne.fit_transform(
        torch.vstack((
            movie_embs,
            user_embs,
        )).detach().numpy())
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(emb_reduced[:movie_embs.size(0),
                            0], emb_reduced[:movie_embs.size(0), 1], alpha=0.1,
                label='All Movies')

    movie_path = osp.join(path, './raw/ml-latest-small/movies.csv')

    def load_node_csv(path, index_col):
        df = pd.read_csv(path, index_col=index_col)
        return df

    movie_id_to_name = load_node_csv(movie_path, index_col='movieId')
    val_pos_items = get_user_positive_items(val_data['user',
                                                     'movie'].edge_index)
    train_pos_items = get_user_positive_items(train_data['user',
                                                         'movie'].edge_index)
    displayed_users = 0
    title_text = "Displaying Train and Pred Movies for users: "

    for user_i, (val_user, gt) in enumerate(val_pos_items.items()):
        if val_user not in users_list:
            continue
        try:
            try:
                train_gt = train_pos_items[val_user]
                plt.scatter(emb_reduced[train_gt, 0], emb_reduced[train_gt, 1],
                            marker='+', label='Train Movies' + str(val_user))
                for i in train_gt:
                    plt.annotate(
                        movie_id_to_name.iloc[i]['title'],
                        xy=(emb_reduced[i, 0], emb_reduced[i, 1]),
                        xytext=(emb_reduced[i, 0] + 2, emb_reduced[i, 1]),
                        fontsize=3, horizontalalignment='left', arrowprops={
                            'arrowstyle': '->',
                            'color': '0.5',
                            'shrinkA': 5,
                            'shrinkB': 5,
                            'connectionstyle':
                            "angle,angleA=-90,angleB=180,rad=0"
                        })
            except KeyError:
                pass
            val_gt = val_pos_items[val_user]
            plt.scatter(emb_reduced[val_gt, 0], emb_reduced[val_gt, 1],
                        marker='+', label='GT (Val) Movies' + str(val_user))
            for i in val_gt:
                plt.annotate(
                    movie_id_to_name.iloc[i]['title'],
                    xy=(emb_reduced[i, 0], emb_reduced[i, 1]),
                    xytext=(emb_reduced[i, 0] + 2, emb_reduced[i, 1]),
                    fontsize=3, horizontalalignment='left', arrowprops={
                        'arrowstyle': '->',
                        'color': '0.5',
                        'shrinkA': 5,
                        'shrinkB': 5,
                        'connectionstyle': "angle,angleA=-90,angleB=180,rad=0"
                    })

            plt.scatter(emb_reduced[real_recs[val_user],
                                    0], emb_reduced[real_recs[val_user], 1],
                        marker='x', label='Pred Movies ' + str(val_user))
            for i in real_recs[val_user]:
                plt.annotate(
                    movie_id_to_name.iloc[i]['title'],
                    (emb_reduced[i, 0], emb_reduced[i, 1]),
                    xytext=(emb_reduced[i, 0] - 2, emb_reduced[i, 1]),
                    fontsize=3, horizontalalignment='right', arrowprops={
                        'arrowstyle': '->',
                        'color': '0.5',
                        'shrinkA': 5,
                        'shrinkB': 5,
                        'connectionstyle': "angle,angleA=-90,angleB=180,rad=0"
                    })
            displayed_users += 1
            title_text += str(val_user) + ' '
        except KeyError:
            pass

    plt.legend(loc='upper left', ncol=3, fontsize=10)
    plt.title(title_text)
    plt.savefig('./fig-' + str(epoch) + '.png')
    plt.clf()
    plt.close()


def train(train_dl):
    model.train()

    for step, batch in enumerate(train_dl):
        optimizer.zero_grad()
        pred = model(batch['user'].n_id, batch['movie'].n_id, batch.x_dict,
                     batch.edge_index_dict, batch['user',
                                                  'movie'].edge_label_index)
        target = batch['user', 'movie'].edge_label
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
    return float(loss)


def compute_metrics(real_recs, inference_dataloader, k):
    # Convert real_recs from a list to tensor
    real_recs_list = []
    [real_recs_list.append(rec) for id, rec in real_recs.items()]
    rec_tensor = torch.tensor(real_recs_list)

    precision = LinkPredPrecision(k)
    ndcg = LinkPredNDCG(k)

    metrics = {}
    for batch in inference_dataloader:
        node_id = batch['user'].n_id
        precision.update(rec_tensor[node_id], batch['user',
                                                    'movie'].edge_index)
        ndcg.update(rec_tensor[node_id], batch['user', 'movie'].edge_index)

    metrics['precision'] = precision.compute()
    metrics['ndcg'] = ndcg.compute()

    return metrics


eval_batch_size = 256

train_node_dataloader = NeighborLoader(
    train_data,
    num_neighbors=[-1] * 2,
    input_nodes=('user', torch.arange(train_data['user'].num_nodes)),
    shuffle=False,
    batch_size=eval_batch_size,
)
val_node_dataloader = NeighborLoader(
    val_data,
    num_neighbors=[-1] * 2,
    input_nodes=('user', torch.arange(val_data['user'].num_nodes)),
    shuffle=False,
    batch_size=eval_batch_size,
)
test_node_dataloader = NeighborLoader(
    test_data,
    num_neighbors=[-1] * 2,
    input_nodes=('user', torch.arange(test_data['user'].num_nodes)),
    shuffle=False,
    batch_size=eval_batch_size,
)

EPOCHS = args.epochs

for epoch in tqdm(range(0, EPOCHS)):
    loss = train(train_dl=train_dataloader)
    real_recs = make_recommendations(model, train_node_dataloader, args.k)
    val_metrics = compute_metrics(real_recs, val_node_dataloader, args.k)
    test_metrics = compute_metrics(real_recs, test_node_dataloader, args.k)

    if args.visualize_emb:
        visualize(model, train_data, val_data, real_recs, epoch)

    # Let's write out the recommendations into a file
    write_recs(real_recs, val_data)

    # Print output
    print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f},'
          f' Val [precision@{args.k} = {val_metrics["precision"]:.3E},'
          f' ndcg@{args.k} = {val_metrics["ndcg"]:.3E}],'
          f' Test [precision@{args.k} = {test_metrics["precision"]:.3E},'
          f' ndcg@{args.k} = {test_metrics["ndcg"]:.3E}]')

# Save the model for good measure
torch.save(model.state_dict(), "./model.bin")
