import math

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z):
        adj = torch.matmul(z, z.t())
        return adj


class GAE(torch.nn.Module):
    def __init__(self, encoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, *args, **kwargs):
        z = self.encoder(*args, **kwargs)
        return self.decoder(z)

    def generate_edge_splits(self, data, val_ratio=0.05, test_ratio=0.1):
        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = math.floor(val_ratio * row.size(0))
        n_t = math.floor(test_ratio * row.size(0))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_edge_index = torch.stack([r, c], dim=0)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask[np.triu_indices(data.num_nodes, k=1)] = 1
        neg_adj_mask[row, col] = 0
        neg_row, neg_col = neg_adj_mask.nonzero().t()

        perm = torch.randperm(neg_row.size(0))
        perm = perm[:n_v + n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        r, c = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([r, c], dim=0)
        neg_adj_mask[r, c] = 0
        r, c = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([r, c], dim=0)
        neg_adj_mask[r, c] = 0
        data.train_neg_adj_mask = neg_adj_mask

        return data

    def reconstruction_loss(self, adj, edge_index, neg_adj_mask):
        row, col = edge_index
        loss_pos = -torch.log(torch.sigmoid(adj[row, col])).mean()
        loss_neg = -torch.log(
            (1 - torch.sigmoid(adj[neg_adj_mask])).clamp(min=1e-8)).mean()
        return loss_pos + loss_neg

    def eval(self, adj, edge_index, neg_edge_index):
        pos_y = adj.new_ones(edge_index.size(1))
        neg_y = adj.new_zeros(neg_edge_index.size(1))

        adj = torch.sigmoid(adj.detach())
        pos_pred = adj[edge_index[0], edge_index[1]]
        neg_pred = adj[neg_edge_index[0], neg_edge_index[1]]

        y = torch.cat([pos_y, neg_y], dim=0).cpu()
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu()

        roc_score = roc_auc_score(y, pred)
        ap_score = average_precision_score(y, pred)

        return roc_score, ap_score
