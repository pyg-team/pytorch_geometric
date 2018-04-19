import itertools

import torch
from torch_geometric.utils import Data


def collate_to_set(data_list):
    keys = data_list[0].keys
    dataset = {key: [] for key in keys}
    slices = {key: [0] for key in keys}

    for data, key in itertools.product(data_list, keys):
        dataset[key].append(data[key])
        slices[key].append(slices[key][-1] + data[key].size(0))

    for key in keys:
        dataset[key] = torch.cat(dataset[key], dim=0)
        slices[key] = torch.LongTensor(slices[key])

    return Data.from_dict(dataset), slices


def collate_to_batch(data_list):
    keys = data_list[0].keys
    batch = {key: [] for key in keys}

    for data, key in itertools.product(data_list, keys):
        batch[key].append(data[key])

    batch['batch'] = []
    for idx, data in enumerate(data_list):
        batch['batch'].append(torch.LongTensor(data.num_nodes).fill_(idx))

    for key in batch.keys():
        batch[key] = torch.cat(batch[key], dim=0)

    if 'edge_index' in keys:
        batch['edge_index'] = batch['edge_index'].t().contiguous()

    return Data.from_dict(batch)
