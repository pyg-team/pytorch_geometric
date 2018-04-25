import itertools

import torch
from torch_geometric.data import Data


def collate_to_set(data_list):
    keys = data_list[0].keys()
    dataset = {key: [] for key in keys}
    slices = {key: [0] for key in keys}
    dims = {key: 0 for key in keys}
    dims['edge_index'] = 1

    for data, key in itertools.product(data_list, keys):
        dataset[key].append(data[key])
        slices[key].append(slices[key][-1] + data[key].size(dims[key]))

    for key in keys:
        dataset[key] = torch.cat(dataset[key], dim=dims[key])
        slices[key] = torch.LongTensor(slices[key])

    return Data.from_dict(dataset), slices


def collate_to_batch(data_list):
    keys = data_list[0].keys()
    batch = {key: [] for key in keys}
    dims = {key: 0 for key in keys}
    dims['edge_index'] = 1

    for data, key in itertools.product(data_list, keys):
        batch[key].append(data[key])

    batch['batch'] = []
    dims['batch'] = 0
    for idx, data in enumerate(data_list):
        batch['batch'].append(torch.LongTensor(data.num_nodes).fill_(idx))

    for key in batch.keys():
        batch[key] = torch.cat(batch[key], dim=dims[key])

    # TODO: Add index to edge indices
    if 'edge_index' in keys:
        pass

    # TODO: Add index to edge indices
    if 'face' in keys:
        pass

    return Data.from_dict(batch)
