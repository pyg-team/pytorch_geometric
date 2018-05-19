from itertools import repeat, product

import torch
from torch_geometric.data import Dataset, Data


class InMemoryDataset(Dataset):
    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __init__(self, root, transform=None):
        super(InMemoryDataset, self).__init__(root, transform)
        self.dataset, self.slices = None, None

    def __len__(self):
        return self.slices[list(self.slices.keys())[0]].size(0) - 1

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(idx)
            return data if self.transform is None else self.transform(data)
        elif isinstance(idx, slice):
            return self.split(range(*idx.indices(len(self))))
        elif isinstance(idx, torch.LongTensor):
            return self.split(idx)
        elif isinstance(idx, torch.ByteTensor):
            return self.split(idx.nonzero())

        raise IndexError(
            'Only integers, slices (`:`) and long or byte tensors are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def shuffle(self):
        return self.split(torch.randperm(len(self)))

    def get(self, idx):
        data = Data()
        for key in self.dataset.keys:
            item, slices = self.dataset[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.dataset.cat_dim(key)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def split(self, index):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__ = self.__dict__
        copy.dataset, copy.slices = self.collate([self.get(i) for i in index])
        return copy

    def collate(self, data_list):
        keys = data_list[0].keys
        dataset = Data()

        for key in keys:
            dataset[key] = []
        slices = {key: [0] for key in keys}

        for data, key in product(data_list, keys):
            dataset[key].append(data[key])
            s = slices[key][-1] + data[key].size(data.cat_dim(key))
            slices[key].append(s)

        for key in keys:
            dataset[key] = torch.cat(dataset[key], dim=data.cat_dim(key))
            slices[key] = torch.LongTensor(slices[key])

        return dataset, slices
