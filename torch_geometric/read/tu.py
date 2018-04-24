import os.path as osp

import torch
from numpy import bincount
from torch import LongTensor as Long
from torch_geometric.read import read_txt
from torch_geometric.utils import coalesce, one_hot
from torch_geometric.data import Data


def filename(prefix, name, path=None):
    name = '{}_{}.txt'.format(prefix, name)
    return name if path is None else osp.join(path, name)


def cat(*seq):
    seq = [t for t in seq if t is not None]
    seq = [t.unsqueeze(-1) if t.dim() == 1 else t for t in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None


def get_tu_filenames(prefix,
                     graph_indicator=False,
                     graph_attributes=False,
                     graph_labels=False,
                     node_attributes=False,
                     node_labels=False,
                     edge_attributes=False,
                     edge_labels=False):

    names = ['A'] + [key for key, value in locals().items() if value is True]
    return [filename(prefix, name) for name in names]


def read_tu_files(path,
                  prefix,
                  graph_indicator=False,
                  graph_attributes=False,
                  graph_labels=False,
                  node_attributes=False,
                  node_labels=False,
                  edge_attributes=False,
                  edge_labels=False):

    edge_index = read_txt(filename(prefix, 'A', path), out=Long()) - 1
    edge_index, perm = coalesce(edge_index.t())
    edge_index = edge_index.t().contiguous()

    x = tmp1 = tmp2 = None
    if node_attributes:
        tmp1 = read_txt(filename(prefix, 'node_attributes', path))
    if node_labels:
        tmp2 = read_txt(filename(prefix, 'node_labels', path), out=Long()) - 1
        tmp2 = one_hot(tmp2)
    x = cat(tmp1, tmp2)

    edge_attr = tmp1 = tmp2 = None
    if edge_attributes:
        tmp1 = read_txt(filename(prefix, 'edge_attributes', path))[perm]
    if edge_labels:
        tmp2 = read_txt(filename(prefix, 'edge_attributes', path), out=Long())
        tmp2 = one_hot(tmp2 - 1)[perm]
    edge_attr = cat(tmp1, tmp2)

    y = None
    if graph_attributes:  # Regression problem.
        y = read_txt(filename(prefix, 'graph_attributes', path))
    if graph_labels:  # Classification problem.
        y = read_txt(filename(prefix, 'graph_labels', path), out=Long()) - 1

    dataset = Data(x, edge_index, edge_attr, y, None)

    if graph_indicator:
        tmp = read_txt(filename(prefix, 'graph_indicator', path), Long()) - 1
    else:
        tmp = Long(x.size(0)).fill_(0)

    return dataset, compute_slices(dataset, tmp)


def compute_slices(dataset, batch):
    num_nodes = batch.size(0)
    graph_slice = torch.arange(0, batch[-1] + 2, out=Long())

    node_slice = torch.cumsum(Long(bincount(batch)), dim=0)
    node_slice = torch.cat([Long([0]), node_slice], dim=0)

    batch = batch[dataset.edge_index[:, 0]]
    edge_slice = torch.cumsum(Long(bincount(batch)), dim=0)
    edge_slice = torch.cat([Long([0]), edge_slice], dim=0)

    slices = {'edge_index': edge_slice}
    if 'x' in dataset:
        slices['x'] = node_slice
    if 'edge_attr' in dataset:
        slices['edge_attr'] = edge_slice
    if 'y' in dataset:
        y_slice = node_slice if dataset.y.size(0) == num_nodes else graph_slice
        slices['y'] = y_slice

    return slices
