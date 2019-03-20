import math
import random

import torch
from sklearn.metrics import roc_auc_score, average_precision_score


class GAE(torch.nn.Module):
    def __init__(self, encoder):
        super(GAE, self).__init__()
        self.encoder = encoder

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode_all(self, z, sigmoid=False):
        adj = torch.matmul(z, z.t())
        adj = torch.sigmoid(adj) if sigmoid else adj
        return adj

    def decode_for_indices(self, z, edge_index, sigmoid=False):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        value = torch.sigmoid(value) if sigmoid else value
        return value

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
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
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def loss(self, z, pos_edge_index, neg_adj_mask):
        pos_loss = -torch.log(
            self.decode_for_indices(z, pos_edge_index, sigmoid=True)).mean()

        neg_loss = -torch.log(
            (1 - self.decode_all(z, sigmoid=True)[neg_adj_mask]).clamp(
                min=1e-8)).mean()

        return pos_loss + neg_loss

    def eval(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decode_for_indices(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decode_for_indices(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
