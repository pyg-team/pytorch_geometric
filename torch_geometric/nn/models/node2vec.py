import math
import torch
import torch.nn as nn
from torch_cluster import random_walk

from ..inits import reset

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Node2Vec(torch.nn.Module):

    def __init__(self, data, d, p, q, r, l, k, ns):
        super(Node2Vec, self).__init__()
        self.data, self.d, self.p, self.q, self.r, self.l, self.k, self.ns = data, d, p, q, r, l, k, ns
        self.embedding = torch.nn.Embedding(data.num_nodes, d).to(dev)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.embedding)

    # random walks on training nodes
    # returns walks in shape [walks, k]
    # tensor [1,2,3,4] means N(1)={2,3,4}
    # TODO add Parameter p,q to random_walk
    def random_walks(self, train_indices=None):
        if train_indices is None:
            train_indices = torch.randperm(self.data.num_nodes).to(dev)
        row, col = self.data.edge_index.to(dev)
        rw = random_walk(row, col, train_indices.repeat(self.r), self.l)
        walk_length = rw.size(1)
        walks_per_rw = 1 + walk_length - self.k

        walks = rw.new_zeros(walks_per_rw * rw.size(0), self.k)
        iter = 0
        for i in range(rw.size(0)):
            for j in range(walks_per_rw):
                walks[iter] = rw[i][j:self.k + j]
                iter += 1
        return walks

    def neg_loss(self, walks):
        batch_size = walks.size(0)
        start = walks[:, 0]
        rest = walks[:, 1:].contiguous()

        x_start = self.embedding(start).view(batch_size, 1, self.d)
        x_rest = self.embedding(rest.view(-1)).view(batch_size, -1, self.d)
        x_start_pos = x_start.expand(batch_size, x_rest.size(1), self.d)

        x = torch.sum(x_start_pos * x_rest, dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(x) + 1e-8)

        # negative samples
        neg_samples = torch.randint(self.data.num_nodes, (batch_size, self.ns), dtype=torch.long, device=dev)
        x_start_neg = x_start.expand(batch_size, self.ns, self.d)

        x_neg_rest = self.embedding(neg_samples)
        x = torch.sum(x_start_neg * x_neg_rest, dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(x) + 1e-8)

        return pos_loss.mean() + neg_loss.mean()

