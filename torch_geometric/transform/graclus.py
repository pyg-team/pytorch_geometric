from __future__ import division

import torch


def graclus(adj, position, level, rid=None):
    adjs = [adj]
    positions = [position]
    clusters = []
    for _ in range(level):
        n = adj.size(0)
        index = adj._indices()
        row, col = index
        start, end = position[row], position[col]
        dist = end - start
        dist = (dist * dist).sum(1)
        std = dist.sqrt().std()
        std = std * std
        weight = torch.exp(dist / (-2 * std))
        adj_d = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

        cluster, cluster_full, singleton = normalized_cut(adj_d, rid)
        rid = None
        clusters.append(cluster_full)

        # Compute new adjaceny.
        adj = cluster_adj(adj, cluster)
        adjs.append(adj)

        # Compute new positions.
        position = cluster_position(position, cluster_full, singleton)
        positions.append(position)

    # Permute inputs, adjacencies and positions.
    perms = compute_perms(clusters)

    adjs = [perm_adj(adjs[i], perms[i]) for i in range(len(perms))]
    positions = [perm_input(positions[i], perms[i]) for i in range(len(perms))]

    return adjs, positions, perms[0]


class Graclus(object):
    def __init__(self, level=1):
        self.level = level

    def __call__(self, data, rid=None):
        input, adj, position = data

        adjs, positions, perm = graclus(adj, position, self.level, rid)
        input = perm_input(input, perm)

        return input, adjs, positions


def normalized_cut(adj, rid=None):
    n = adj.size(0)
    rid = torch.randperm(n) if rid is None else rid
    cluster = torch.LongTensor(n).fill_(-1)

    row, col = adj._indices()
    weight = adj._values()

    # Fix graphs with self-loops.
    mask = row != col
    row, col, weight = row[mask], col[mask], weight[mask]

    one = weight.new(weight.size(0)).fill_(1)
    degree = 1 / weight.new(n).fill_(0).scatter_add_(0, row, one)
    weight = weight * (degree[row] + degree[col])

    # Sort after weight.
    weight, perm = weight.sort(dim=0, descending=True)
    row = row[perm]
    col = col[perm]

    # Sort after rid.
    _, perm = rid[row].sort()
    row = row[perm]
    col = col[perm]

    # Find cluster values.
    count = 0
    while row.dim() > 0:
        cluster[row[0]] = count
        cluster[col[0]] = count

        mask = (row != row[0]) & (row != col[0])
        mask &= (col != row[0]) & (col != col[0])

        row = row[mask]
        col = col[mask]

        count += 1

    # Append singleton values to the end.
    singleton = cluster == -1
    num_singletons = singleton.sum()
    if num_singletons > 0:
        index = torch.arange(count, count + num_singletons).long()
        cluster[singleton] = index
        cluster_full = torch.cat([cluster, index], dim=0)
        return cluster, cluster_full, singleton
    else:
        return cluster, cluster, singleton


def cluster_adj(adj, cluster):
    n = cluster.max() + 1
    row, col = adj._indices()
    row, col = cluster[row], cluster[col]
    weight = adj._values()
    mask = row != col
    row, col, weight = row[mask], col[mask], weight[mask]
    index = torch.stack([row, col], dim=0)
    adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
    return adj


def cluster_position(pos, cluster, singleton):
    dim = pos.size(1)
    singleton = singleton.repeat(dim).view(dim, -1).t()
    pos = torch.cat([pos, pos[singleton].view(-1, dim)], dim=0)
    n = cluster.max() + 1
    return torch.stack(
        [
            pos.new(n).fill_(0).scatter_add_(0, cluster, pos[:, i]) / 2
            for i in range(dim)
        ],
        dim=1)


def compute_perms(clusters):
    n = clusters[-1].max() + 1

    perm = torch.arange(0, n).long()
    perms = [perm]

    for i in range(len(clusters) - 1, -1, -1):
        cluster = clusters[i]
        max_cluster = cluster.max() + 1

        # Append double fake nodes.
        if max_cluster < n:
            index = torch.arange(max_cluster, n).long().repeat(2)
            cluster = torch.cat([cluster, index], dim=0)

        cluster = perm.sort()[1][cluster]
        _, rid = cluster.sort()
        n *= 2
        perm = torch.arange(0, n).long()[rid]
        perms.append(perm)

    return perms[::-1]


def perm_adj(adj, perm):
    n = perm.size(0)
    row, col = adj._indices()

    _, perm = perm.sort()
    row = perm[row]
    col = perm[col]
    index = torch.stack([row, col], dim=0)

    adj = torch.sparse.FloatTensor(index, adj._values(), torch.Size([n, n]))
    return adj.coalesce()


def perm_input(input, perm):
    n = input.size(0)
    num_fake_nodes = perm.size(0) - n
    size = list(input.size())
    size[0] = num_fake_nodes
    input = torch.cat([input, input.new(*size).fill_(0)], dim=0)
    return input[perm]
