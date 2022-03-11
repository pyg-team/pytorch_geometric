import copy
import math
import os.path as osp

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.nn import Parameter

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import GATConv, GCNConv

EPS = 1e-15
FEATURE_DIM = 64
EMBED_DIM = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/DBLP')
dataset = DBLP(path)
data = dataset[0]
print(data)

# Initialize author node features
author_len = data['author'].num_nodes
data['author'].x = torch.ones(data['author'].num_nodes, FEATURE_DIM)
# Select metapath APCPA and APA as example metapaths.
metapaths = [[("author", "paper"), ("paper", "conference"),
             ("conference", 'paper'), ('paper', 'author')],
             [("author", "paper"), ("paper", "author")]]
data = T.AddMetaPaths(metapaths)(data)


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATEncoder, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads)
        self.activation = torch.nn.PReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        return self.activation(self.conv(x, edge_index))


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.activation = torch.nn.PReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.conv(x, edge_index))
        return x


class HeteroUnsupervised(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Z = torch.nn.Parameter(torch.Tensor(author_len, out_channels))
        # for discriminator
        self.embed_size = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()

        # for encoders
        # each encoder corresponds to each metapath subgraph,
        # here we used APA and APCPA.
        self.encoders = [GCNEncoder(in_channels, out_channels).to(device)
                         for _ in range(2)]

    def forward(self, x_dict, edge_index_dict):
        target_embeds = x_dict['author']

        edge_index_a = edge_index_dict['author', 'metapath_0', 'author']
        edge_index_b = edge_index_dict['author', 'metapath_1', 'author'] 

        mp_edge_pairs = [edge_index_a, edge_index_b]

        pos_feats = []
        for _ in range(len(mp_edge_pairs)):
            pos_feats.append(copy.deepcopy(target_embeds))
        pos_embeds = []
        neg_embeds = []
        summaries = []
        for encoder, pos_feat, edge_index in \
                zip(self.encoders, pos_feats, mp_edge_pairs):
            # shuffle feature corruption
            shuffle_idx = np.random.permutation(pos_feat.shape[0])
            corrupted_feat = pos_feat[shuffle_idx]

            pos_embed = encoder(pos_feat, edge_index)
            neg_embed = encoder(corrupted_feat, edge_index)
            summary = torch.mean(pos_embed, dim=0)

            pos_embeds.append(pos_embed)
            neg_embeds.append(neg_embed)
            summaries.append(summary)

        return pos_embeds, neg_embeds, summaries

    def embed(self):
        return self.Z

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Z)
        uniform(self.embed_size, self.weight)

    def loss(self, pos_embeds, neg_embeds, summaries):
        total_loss = 0
        for pos_embed, neg_embed, summary in \
                zip(pos_embeds, neg_embeds, summaries):

            pos_loss = -torch.log(
                self.discriminate(pos_embed, summary, sigmoid=True) +
                EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(
                neg_embed, summary, sigmoid=True) + EPS).mean()
            total_loss += (pos_loss + neg_loss)

        pos_mean = torch.stack(pos_embeds).mean(dim=0)
        neg_mean = torch.stack(neg_embeds).mean(dim=0)
        # consensus regularizer
        pos_reg_loss = ((self.Z - pos_mean)**2).sum()
        neg_reg_loss = ((self.Z - neg_mean)**2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        total_loss += reg_loss
        return total_loss


model = HeteroUnsupervised(out_channels=FEATURE_DIM, in_channels=FEATURE_DIM)

data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_embeds, neg_embeds, summaries = \
        model(data.x_dict, data.edge_index_dict)

    # pick out the train embed
    # mask = data['author']['train_mask']
    # pos_embeds = [pos_embed[mask] for pos_embed in pos_embeds]
    # neg_embeds = [neg_embed[mask] for neg_embed in neg_embeds]

    loss = model.loss(pos_embeds, neg_embeds, summaries)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    n_iter = 1000000

    pos_embed = model.embed()
    mask = data['author']['val_mask']
    labels = data['author'].y[mask]

    valid_embed = pos_embed[mask]
    valid_embed = valid_embed.cpu().numpy()
    labels = labels.cpu().numpy()

    train_z, test_z, train_y, test_y = \
        train_test_split(valid_embed, labels, train_size=0.8)

    clf = LogisticRegression(max_iter=n_iter).fit(train_z, train_y)
    val_score = clf.score(test_z, test_y)

    mask = data['author']['test_mask']
    labels = data['author'].y[mask]

    test_embed = pos_embed[mask]
    test_embed = test_embed.cpu().numpy()
    labels = labels.cpu().numpy()

    train_z, test_z, train_y, test_y = \
        train_test_split(test_embed, labels, train_size=0.8)

    clf = LogisticRegression(max_iter=n_iter).fit(train_z, train_y)
    test_score = clf.score(test_z, test_y)

    # accuracy score of author nodes.
    return val_score, test_score


for epoch in range(1, 1000):
    loss = train()
    valid_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},\
         Valid: {test_acc:.4f}, Test: {test_acc:.4f}')
