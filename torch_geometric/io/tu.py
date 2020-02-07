import os
import os.path as osp
import glob

import pandas
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).sub_(1)
    edge_index = edge_index.t().contiguous()

    batch = read_file(folder, prefix, 'graph_indicator', torch.long).sub_(1)

    node_attrs = node_labels = None
    if 'node_attributes' in names:
        node_attrs = read_file(folder, prefix, 'node_attributes', torch.float)
        if node_attrs.dim() == 1:
            node_attrs.unsqueeze(-1)
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels.sub_(node_labels.min(dim=0)[0])
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attrs, node_labels])

    edge_attrs = edge_labels = None
    if 'edge_attributes' in names:
        edge_attrs = read_file(folder, prefix, 'edge_attributes', torch.float)
        if edge_attrs.dim() == 1:
            edge_attrs.unsqueeze(-1)
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels.sub_(edge_labels.min(dim=0)[0])
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attrs, edge_labels])

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    adj = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_attr,
                       sparse_sizes=torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce().remove_diag().fill_cache_()

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes', torch.float)
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        # Transform graph labels to consecutive indices.
        _, y = y.unique(sorted=True, return_inverse=True)

    batch_ptr = torch.ops.torch_sparse.ind2ptr(batch, y.size(0))

    data_list = []
    for i in range(y.size(0)):
        start = batch_ptr[i]
        length = batch_ptr[i + 1] - start

        data = Data()
        data.adj = adj.__narrow_diag__((start, start), (length, length))
        if x is not None:
            data.x = x.narrow(0, start, length)
        data.y = y[i].item() if y[i].numel() == 1 else y[i]

        data_list.append(data)

    num_node_attributes = node_attrs.size(-1) if node_attrs is not None else 0
    num_edge_attributes = edge_attrs.size(-1) if edge_attrs is not None else 0

    return data_list, num_node_attributes, num_edge_attributes


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    data_frame = pandas.read_csv(path, sep=',', header=None)
    return torch.from_numpy(data_frame.to_numpy()).to(dtype).squeeze(-1)


def cat(seq):
    seq = [item for item in seq if item is not None]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None
