import numpy as np
from torch.nn import Parameter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os.path as osp
import math
import torch
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv

EPS = 1e-15

path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets/DBLP')
dataset = DBLP(path)
data = dataset[0]
print(data)

# We initialize conference node features with a single feature.
data['conference'].x = torch.ones(data['conference'].num_nodes, 1)


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


class HeteroUnsupervised(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        # for discriminator
        self.embed_size = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()

        # for encoder
        self.encoder = GATEncoder(hidden_channels, out_channels, heads=1)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, xx_dict, edge_index_dict):
        # using meta path author-paper-author
        x_dict = {}
        for conv in self.convs:
            x_dict = conv(xx_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        target_embeds = x_dict['author']

        edge_index_a = edge_index_dict[('author', 'to', 'paper')]
        edge_index_b = edge_index_dict[('paper', 'to', 'author')]

        APA_edge_pair= \
            self.convert_to_APA_edge_index(edge_index_a, edge_index_b)
        # please mannually preprocess corresponding metapath edge pair
        # this is just an example and NOT real ACPCA edge pair
        ACPCA_edge_pair=APA_edge_pair[:, -5]

        mp_edge_pairs=[APA_edge_pair,ACPCA_edge_pair]

        pos_feats = [target_embeds for _ in range(len(mp_edge_pairs))]
        pos_embeds=[]
        neg_embeds=[]
        summaries=[]
        for pos_feat, edge_index in zip(pos_feats, mp_edge_pairs):
            # shuffle feature corruption
            shuffle_idx = np.random.permutation(pos_feat.shape[0])
            corrupted_feat = pos_feat[shuffle_idx]

            pos_embed = self.encoder(pos_feat, edge_index)
            neg_embed = self.encoder(corrupted_feat, edge_index)
            summary = torch.mean(pos_embed, dim=0)

            pos_embeds.append(pos_embed)
            neg_embeds.append(neg_embed)
            summaries.append(summary)

        return pos_embeds, neg_embeds, summaries

    def embed(self, xx_dict, edge_index_dict):
        # using meta path author-paper-author
        x_dict = {}
        for conv in self.convs:
            x_dict = conv(xx_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        target_embeds = x_dict['author']

        edge_index_a = edge_index_dict[('author', 'to', 'paper')]
        edge_index_b = edge_index_dict[('paper', 'to', 'author')]

        APA_edge_pair= \
            self.convert_to_APA_edge_index(edge_index_a, edge_index_b)
        # Please mannually preprocess corresponding metapath edge pair
        # This is just an example and NOT real ACPCA edge pair
        ACPCA_edge_pair = APA_edge_pair[:,-5]

        mp_edge_pairs=[APA_edge_pair, ACPCA_edge_pair]

        pos_feats = [target_embeds for _ in range(len(mp_edge_pairs))]
        pos_embeds=[]
        for pos_feat, edge_index in zip(pos_feats, mp_edge_pairs):
            pos_embed = self.encoder(pos_feat, edge_index)
            pos_embeds.append(pos_embed)

        # mean aggeragation
        final_embed = sum(pos_embeds) / len(mp_edge_pairs)
        return final_embed

    def convert_to_APA_edge_index(self, edge_index_a, edge_index_b):
        paper2author_dict = dict(edge_index_b.T.cpu().numpy())
        target_edge_index = []
        for idx_a, idx_b in edge_index_a.T:
            idx_b = idx_b.item()
            idx_a = idx_a.item()
            if paper2author_dict.get(idx_b) is not None:
                target_edge_index.append((idx_a, paper2author_dict[idx_b]))
        target_edge_index = torch.tensor(target_edge_index, dtype=torch.long)
        target_edge_index = target_edge_index.T

        return target_edge_index.to(device)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        uniform(self.embed_size, self.weight)

    def loss(self, pos_embeds, neg_embeds, summaries):
        total_loss=0
        for pos_embed, neg_embed, summary in \
            zip(pos_embeds, neg_embeds, summaries):

            pos_loss = -torch.log(
                self.discriminate(pos_embed, summary, sigmoid=True) + EPS
            ).mean()
            neg_loss = -torch.log(
                1 - self.discriminate(neg_embed, summary, sigmoid=True) + EPS
            ).mean()
            total_loss += (pos_loss + neg_loss)

        return total_loss


model = HeteroUnsupervised(
    data.metadata(),
    out_channels=64,
    hidden_channels=64,
    num_layers=2
)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_embeds, neg_embeds, summaries = \
        model(data.x_dict, data.edge_index_dict)

    # pick out the train embed
    mask = data['author']['train_mask']
    pos_embeds = [pos_embed[mask] for pos_embed in pos_embeds]
    neg_embeds = [neg_embed[mask] for neg_embed in neg_embeds]

    loss = model.loss(pos_embeds, neg_embeds, summaries)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pos_embed = model.embed(data.x_dict, data.edge_index_dict)
    mask = data['author']['val_mask']
    labels = data['author'].y[mask]

    valid_embed = pos_embed[mask]

    train_z, test_z, train_y, test_y = \
        train_test_split(valid_embed, labels, train_size=0.8)

    clf = LogisticRegression(solver='lbfgs').fit(train_z, train_y)
    val_score = clf.score(test_z, test_y)

    mask = data['author']['test_mask']
    labels = data['author'].y[mask]

    test_embed = pos_embed[mask]

    train_z, test_z, train_y, test_y = \
        train_test_split(test_embed, labels, train_size=0.8)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000000)\
                .fit(train_z, train_y)
    test_score = clf.score(test_z, test_y)

    # accuracy score of author nodes.
    return val_score, test_score


for epoch in range(1, 101):
    loss = train()
    valid_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},\
         Valid: {test_acc:.4f}, Test: {test_acc:.4f}')
