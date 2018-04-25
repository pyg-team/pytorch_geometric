from torch_geometric.data import Data, collate_to_set


def data_from_set(dataset, slices, i):
    data = Data()
    keys = dataset.keys()

    if dataset.edge_index is not None:
        s = slices['edge_index']
        data.edge_index = dataset.edge_index[:, s[i]:s[i + 1]]
        keys.remove('edge_index')

    for key in keys:
        data[key] = dataset[key][slices[key][i]:slices[key][i + 1]]

    return data


def split_set(dataset, slices, split):
    data_list = [data_from_set(dataset, slices, i) for i in split]
    return collate_to_set(data_list)
