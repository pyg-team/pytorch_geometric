import os
import os.path as osp

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.io import fs, read_txt_array
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
    files = fs.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
        if node_attributes.dim() == 1:
            node_attributes = node_attributes.unsqueeze(-1)

    node_labels = torch.empty((batch.size(0), 0))
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [one_hot(x) for x in node_labels]
        if len(node_labels) == 1:
            node_labels = node_labels[0]
        else:
            node_labels = torch.cat(node_labels, dim=-1)

    edge_attributes = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
        if edge_attributes.dim() == 1:
            edge_attributes = edge_attributes.unsqueeze(-1)

    edge_labels = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [one_hot(e) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_labels = edge_labels[0]
        else:
            edge_labels = torch.cat(edge_labels, dim=-1)

    x = cat([node_attributes, node_labels])
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_node_labels': node_labels.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_labels': edge_labels.size(-1),
    }

    return data, slices, sizes


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = cumsum(torch.from_numpy(np.bincount(batch)))

    row, _ = data.edge_index
    edge_slice = cumsum(torch.from_numpy(np.bincount(batch[row])))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices
