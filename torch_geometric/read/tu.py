import os.path as osp

import torch
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
        tmp2 = to_tensor(path, prefix, 'node_labels', LongTensor()) - 1
        tmp2 = one_hot(tmp2)
    x = cat(tmp1, tmp2)

    edge_attr = tmp1 = tmp2 = None
    if edge_attributes:
        tmp1 = to_tensor(path, prefix, 'edge_attributes')[perm]
    if edge_labels:
        tmp2 = to_tensor(path, prefix, 'edge_labels', LongTensor())[perm] - 1
        tmp2 = one_hot(tmp2)
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
        tmp = torch.LongTensor(x.size(0)).fill_(0)

    return dataset, compute_slices(dataset, tmp)


def compute_slices(dataset, batch):
    print(batch)
    print(dataset.edge_index)
    pass
