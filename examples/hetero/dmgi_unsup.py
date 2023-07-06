# An implementation of "Unsupervised Attributed Multiplex Network
# Embedding" (DMGI) for unsupervised learning on  heterogeneous graphs:
# * Paper: <https://arxiv.org/abs/1911.06750> (AAAI 2020)

import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam

import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB
from torch_geometric.nn import GCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
dataset = IMDB(path)

metapaths = [
    [('movie', 'actor'), ('actor', 'movie')],  # MAM
    [('movie', 'director'), ('director', 'movie')],  # MDM
]
data = T.AddMetaPaths(metapaths, drop_orig_edge_types=True)(dataset[0])


class DMGI(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)])
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.empty(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)

        return loss


model = DMGI(data['movie'].num_nodes, data['movie'].x.size(-1),
             out_channels=64, num_relations=len(data.edge_types))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)


def train():
    model.train()
    optimizer.zero_grad()
    x = data['movie'].x
    edge_indices = data.edge_index_dict.values()
    pos_hs, neg_hs, summaries = model(x, edge_indices)
    loss = model.loss(pos_hs, neg_hs, summaries)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    train_emb = model.Z[data['movie'].train_mask].cpu()
    val_emb = model.Z[data['movie'].val_mask].cpu()
    test_emb = model.Z[data['movie'].test_mask].cpu()

    train_y = data['movie'].y[data['movie'].train_mask].cpu()
    val_y = data['movie'].y[data['movie'].val_mask].cpu()
    test_y = data['movie'].y[data['movie'].test_mask].cpu()

    clf = LogisticRegression().fit(train_emb, train_y)
    return clf.score(val_emb, val_y), clf.score(test_emb, test_y)


for epoch in range(1, 1001):
    loss = train()
    if epoch % 50 == 0:
        val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
