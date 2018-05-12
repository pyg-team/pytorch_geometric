from itertools import repeat
from torch_geometric.data import Data, collate_to_set


def data_from_set(dataset, slices, i):
    data = Data()
    for key in dataset.keys():
        s = list(repeat(slice(None), dataset[key].dim()))
        s[dataset.cat_dim(key)] = slice(slices[key][i], slices[key][i + 1])
        data[key] = dataset[key][s]
    return data


def split_set(dataset, slices, split):
    data_list = [data_from_set(dataset, slices, i) for i in split]
    return collate_to_set(data_list)
