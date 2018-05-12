import itertools

import torch
from torch_geometric.data import Data


def collate_to_set(data_list):
    keys = data_list[0].keys()
    data_dict = {key: [] for key in keys}
    slices = {key: [0] for key in keys}

    for data, key in itertools.product(data_list, keys):
        data_dict[key].append(data[key])
        slices[key].append(slices[key][-1] + data[key].size(data.cat_dim(key)))

    for key in keys:
        data_dict[key] = torch.cat(data_dict[key], dim=data.cat_dim(key))
        slices[key] = torch.LongTensor(slices[key])

    return Data.from_dict(data_dict), slices


def collate_to_batch(data_list):
    keys = data_list[0].keys()
    batch = {key: [] for key in keys}
    batch['batch'] = []

    cumsum = 0
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        batch['batch'].append(torch.full((num_nodes, ), i, dtype=torch.long))
        for key in keys:
            t = data[key] + cumsum if data.should_cumsum(key) else data[key]
            batch[key].append(t)
        cumsum += num_nodes

    for key in keys:
        batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key))
    batch['batch'] = torch.cat(batch['batch'], dim=0)

    return Data.from_dict(batch).contiguous()
