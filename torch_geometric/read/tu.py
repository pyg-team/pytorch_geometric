import os.path as osp

import torch
from numpy import bincount
from torch import LongTensor
from torch_geometric.utils import coalesce, one_hot
from torch_geometric.data import Data


def filename(prefix, name):
    return '{}_{}.txt'.format(prefix, name)


def to_tensor(path, prefix, name, out=None):
    with open(osp.join(path, filename(prefix, name)), 'r') as f:
        content = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(',')] for line in content]

    src = torch.Tensor(content).squeeze()
    return src if out is None else out.resize_(src.size()).copy_(src)


def cat(*seq):
    seq = [t for t in seq if t is not None]
    seq = [t.unsqueeze(-1) if t.dim() == 1 else t for t in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None


def tu_raw_files(prefix,
                 graph_indicator=False,
                 graph_attributes=False,
                 graph_labels=False,
                 node_attributes=False,
                 node_labels=False,
                 edge_attributes=False,
                 edge_labels=False):

    names = ['A'] + [key for key, value in locals().items() if value is True]
    return [filename(prefix, name) for name in names]


def tu_files_to_set(path,
                    prefix,
                    graph_indicator=False,
                    graph_attributes=False,
                    graph_labels=False,
                    node_attributes=False,
                    node_labels=False,
                    edge_attributes=False,
                    edge_labels=False):

    edge_index = to_tensor(path, prefix, 'A', out=LongTensor()) - 1
    edge_index, perm = coalesce(edge_index.t())
    edge_index = edge_index.t().contiguous()

    x = tmp1 = tmp2 = None
    if node_attributes:
        tmp1 = to_tensor(path, prefix, 'node_attributes')
    if node_labels:
        tmp2 = to_tensor(path, prefix, 'node_labels', out=LongTensor()) - 1
        tmp2 = one_hot(tmp2)
    x = cat(tmp1, tmp2)

    edge_attr = tmp1 = tmp2 = None
    if edge_attributes:
        tmp1 = to_tensor(path, prefix, 'edge_attributes')[perm]
    if edge_labels:
        tmp2 = to_tensor(path, prefix, 'edge_labels', out=LongTensor()) - 1
        tmp2 = one_hot(tmp2)[perm]
    edge_attr = cat(tmp1, tmp2)

    y = None
    if graph_attributes:  # Regression problem.
        y = to_tensor(path, prefix, 'graph_attributes')
    if graph_labels:  # Classification problem.
        y = to_tensor(path, prefix, 'graph_labels', out=LongTensor()) - 1

    dataset = Data(x, edge_index, edge_attr, y, None)

    if graph_indicator:
        tmp = to_tensor(path, prefix, 'graph_indicator', out=LongTensor()) - 1
    else:
        tmp = LongTensor(x.size(0)).fill_(0)

    return dataset, compute_slices(dataset, tmp)


def compute_slices(dataset, batch):
    num_nodes = batch.size(0)
    graph_slice = torch.arange(0, batch[-1] + 2, out=LongTensor())

    node_slice = torch.cumsum(LongTensor(bincount(batch)), dim=0)
    node_slice = torch.cat([LongTensor([0]), node_slice], dim=0)

    batch = batch[dataset.edge_index[:, 0]]
    edge_slice = torch.cumsum(LongTensor(bincount(batch)), dim=0)
    edge_slice = torch.cat([LongTensor([0]), edge_slice], dim=0)

    slices = {'edge_index': edge_slice}
    if 'x' in dataset:
        slices['x'] = node_slice
    if 'edge_attr' in dataset:
        slices['edge_attr'] = edge_slice
    if 'y' in dataset:
        y_slice = node_slice if dataset.y.size(0) == num_nodes else graph_slice
        slices['y'] = y_slice

    return slices
