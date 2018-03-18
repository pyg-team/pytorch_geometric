import os
import os.path as osp

from ..dataset import Set

import torch
import numpy as np


def to_one_hot(tensor):
    tensor -= tensor.min()
    max = tensor.max(dim=0)[0] + 1
    D = max.sum()
    out = torch.FloatTensor(tensor.size(0), D).fill_(0)
    offset = tensor.new(max.size()).fill_(0)
    for i in range(1, offset.size(0)):
        offset[i] = offset[i - 1] + max[i - 1]

    tensor += offset
    offset = torch.arange(0, tensor.size(0) * D, D, out=torch.LongTensor())
    tensor += offset.view(-1, 1)

    out.view(-1)[tensor.view(-1)] = 1
    return out.view(-1, D)


def tu_files(prefix,
             graph_indicator=False,
             graph_attributes=False,
             graph_labels=False,
             node_attributes=False,
             node_labels=False,
             edge_attributes=False,
             edge_labels=False,
             graph_sets=False):
    def func(x):
        return ['{}_{}.txt'.format(prefix, x)]

    files = []
    files += func('A')
    files += func('graph_indicator') if graph_indicator else []
    files += func('graph_attributes') if graph_attributes else []
    files += func('graph_labels') if graph_labels else []
    files += func('node_attributes') if node_attributes else []
    files += func('node_labels') if node_labels else []
    files += func('edge_attributes') if edge_attributes else []
    files += func('edge_labels') if edge_labels else []
    files += func('graph_sets') if graph_sets else []
    return files


def read_tu_set(dir,
                prefix,
                graph_indicator=False,
                graph_attributes=False,
                graph_labels=False,
                node_attributes=False,
                node_labels=False,
                edge_attributes=False,
                edge_labels=False):

    index = read_tu_index(dir, prefix)
    input = read_tu_x(dir, prefix, 'node', node_attributes, node_labels)
    weight = read_tu_x(dir, prefix, 'edge', edge_attributes, edge_labels)
    data = read_tu_slice(dir, prefix, index, graph_indicator)
    index, slice, index_slice = data

    if graph_labels is True and graph_attributes is False:
        f = 'graph_labels'
        target = read_tu_file(dir, prefix, f, out=torch.LongTensor())
        target = target.squeeze()
        target -= target.min()
    else:
        f = 'graph'
        target = read_tu_x(dir, prefix, f, graph_attributes, graph_labels)

    return Set(input, None, index, weight, target, None, slice, index_slice)


def read_tu_file(dir, prefix, name, out=None):
    with open(osp.join(dir, '{}_{}.txt'.format(prefix, name)), 'r') as f:
        lines = f.read().split('\n')[:-1]
        tensor = [[float(x.strip()) for x in l.split(',')] for l in lines]
    tensor = torch.FloatTensor(tensor)
    return tensor if out is None else tensor.type_as(out)


def read_tu_index(dir, prefix):
    index = read_tu_file(dir, prefix, 'A', out=torch.LongTensor()).t()
    index -= index.min()
    n = index.max()
    weight = torch.ByteTensor(index.size(1))
    adj = torch.sparse.ByteTensor(index, weight, torch.Size([n, n])).coalesce()
    return adj._indices()


def read_tu_x(dir, prefix, name, attributes=False, labels=False):
    if attributes is False and labels is False:
        return None

    x = []
    if attributes is True:
        f = '{}_attributes'.format(name)
        x += [read_tu_file(dir, prefix, f)]

    if labels is True:
        f = '{}_labels'.format(name)
        x += [to_one_hot(read_tu_file(dir, prefix, f, out=torch.LongTensor()))]

    return x[0] if len(x) == 1 else torch.cat(x, dim=1)


def read_tu_slice(dir, prefix, index, graph_indicator=False):
    if graph_indicator is True:
        f = 'graph_indicator'
        batch = read_tu_file(dir, prefix, f, out=torch.LongTensor()) - 1
        batch = batch.squeeze()

        slice, cur_i = [0], 0
        for idx, i in enumerate(batch):
            if cur_i != i:
                slice.append(idx)
                cur_i += 1
        slice.append(batch.size(0))

        index_slice, cur_i, cur_off = [0], 0, 0
        off = torch.LongTensor(index.size(1))
        for idx, i in enumerate(index[0]):
            if cur_i != batch[i]:
                index_slice.append(idx)
                cur_i += 1
                cur_off = index[0, idx]
            off[idx] = cur_off
        index_slice.append(index.size(1))
        index -= off
    else:
        slice, index_slice = [0, index.max() + 1], [0, index.size(1)]

    return index, torch.LongTensor(slice), torch.LongTensor(index_slice)


# OLD VERSION -----------------------------------------------------------------


def read_file(dir, prefix, name):
    path = os.path.join(dir, '{}_{}.txt'.format(prefix, name))
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(', ')] for line in lines]
    return torch.FloatTensor(content).squeeze()


def read_adj(dir, prefix):
    index = read_file(dir, prefix, 'A')
    index = index.t().long() - 1
    row, col = index
    row, perm = row.sort()
    col = col[perm]
    index = torch.stack([row, col], dim=0)
    indicator = read_file(dir, prefix, 'graph_indicator').long() - 1

    index_slice = index.new(indicator.max() + 2)
    index_slice[0] = 0
    index_slice[-1] = index.size(1)
    curr_graph = indicator[0]

    for i in range(index.size(1)):
        row = index[0, i]
        if indicator[row] > curr_graph:
            j = index_slice[curr_graph]
            index[:, j:i] -= index[:, j:i].min()
            curr_graph += 1
            index_slice[curr_graph] = i
    index[:, index_slice[curr_graph]:] -= index[:, index_slice[
        curr_graph]:].min()

    return index, index_slice


def read_slice(dir, prefix):
    indicator = read_file(dir, prefix, 'graph_indicator').squeeze().long() - 1
    slice = np.cumsum(np.bincount(indicator.numpy()))
    slice = torch.from_numpy(slice)
    return torch.cat([torch.LongTensor([0]), slice], dim=0)
