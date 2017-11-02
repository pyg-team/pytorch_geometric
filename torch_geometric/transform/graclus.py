from __future__ import division

import torch


class Graclus(object):
    def __init__(self, level=1):
        self.level = level

    def __call__(self, data, rid=None):
        input, adj, position = data

        adjs = [adj]
        positions = [position]
        clusters = []
        for _ in range(self.level):
            print('drin')
            # TODO: if position is not None, compute adj_dist
            cluster, cluster_full, singleton = normalized_cut(adj, rid)
            print(cluster_full.size(), cluster.size())
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
        positions = [
            perm_input(positions[i], perms[i]) for i in range(len(perms))
        ]
        input = perm_input(input, perms[0])

        return input, adjs, positions


def normalized_cut(adj, rid=None):
    n = adj.size(0)
    rid = torch.randperm(n) if rid is None else rid
    cluster = torch.LongTensor(n).fill_(-1)

    row, col = adj._indices()
    weight = adj._values()
    degree = 1 / weight.new(n).fill_(0).scatter_add_(0, row, weight)
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
        print(row[0], col[0])
        if row[0] == col[0]:
            print('das darf nicht sein')
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
    index = torch.stack([row, col], dim=0)
    weight = adj._values()
    weight[row == col] = 0
    adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
    return adj.coalesce()


def cluster_position(pos, cluster, singleton):
    dim = pos.size(1)
    singleton = singleton.repeat(dim).view(dim, -1).t()
    pos = torch.cat([pos, pos[singleton].view(-1, dim)], dim=0)
    n = cluster.max() + 1
    pos = torch.stack(
        [
            pos.new(n).fill_(0).scatter_add_(0, cluster, pos[:, i]) / 2
            for i in range(dim)
        ],
        dim=1)

    return pos


def compute_perms(clusters):
    n = clusters[-1].max() + 1

    perm = torch.arange(0, n).long()
    perms = [perm]

    for i in range(len(clusters) - 1, -1, -1):
        cluster = clusters[i]
        max_cluster = cluster.max() + 1
        # print('max_cluster', max_cluster, 'n', n, cluster.size())
        if max_cluster < n:
            index = torch.arange(max_cluster, n).long().repeat(2)
            cluster = torch.cat([cluster, index], dim=0)

        # print(perm.max(), perm.size())
        # print(cluster.max(), cluster.size())
        # cluster = perm.sort()[1][cluster]
        # rid = cluster.sort()[1]
        n *= 2
        perm = torch.arange(0, n).long()
        perms.append(perm)

    return perms[::-1]


def perm_adj(adj, perm):
    n = perm.size(0)
    row, col = adj._indices()

    perm, _ = perm.sort()
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
