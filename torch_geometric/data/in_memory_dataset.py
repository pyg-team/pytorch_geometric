import torch
from torch_sparse import SparseTensor, SparseStorage
from torch_geometric.data import Dataset


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into memory.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return []

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return []

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(self, root=None, transform=None, pre_transform=None,
                 pre_filter=None):
        self.__cached_data__ = {}
        super(InMemoryDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        if self.data_list[0].y.dim() > 1:
            return self.data_list[0].y.size(1)

        max_num_classes = 0
        for data in self.data_list:
            max_num_classes = max(int(data.y.max()) + 1, max_num_classes)
        return max_num_classes

    def len(self):
        return self.data['__length__']

    def get(self, idx):
        cached_data = self.__cached_data__.get(idx)
        if cached_data is not None:
            return cached_data

        data = self.data['__data_class__']()

        for key in self.slices.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                dim = data.__cat_dim__(key, item)
                start, end = slices[idx], slices[idx + 1]
                data[key] = item.narrow(dim, start, end - start)
            elif (isinstance(item, dict)
                  and item.get('__data_class__') == SparseTensor):
                storage = SparseStorage.empty()
                for k in slices.keys():
                    setattr(storage, k,
                            item[k][slices[k][idx]:slices[k][idx + 1]])
                storage._sparse_sizes = item['_sparse_sizes'][idx]
                data[key] = item.get('__data_class__').from_storage(storage)
            else:
                start, end = slices[idx], slices[idx + 1]
                if end - start == 1:
                    data[key] = item[idx]
                else:
                    data[key] = item[start:end]

        self.__cached_data__[idx] = data

        return data

    def collate(self, data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        ref = data_list[0]
        keys = [key for key in ref.__dict__.keys() if ref[key] is not None]
        keys = [
            key[2:-2] if key in ref.__private_keys__ else key for key in keys
        ]

        store, ptrs = {}, {}
        for key in keys:
            item = ref[key]
            if isinstance(item, SparseTensor):
                storage = item.storage
                storage_keys = [
                    k for k in storage.__dict__.keys()
                    if getattr(storage, k, None) is not None
                ]
                store[key] = {k: [] for k in storage_keys}
                storage_keys.remove('_sparse_sizes')
                ptrs[key] = {k: [0] for k in storage_keys}
            else:
                store[key] = []
                ptrs[key] = [0]

        for data in data_list:
            for key in keys:
                item = data[key]
                if torch.is_tensor(item):
                    store[key].append(item)
                    ptrs[key].append(ptrs[key][-1] +
                                     item.size(data.__cat_dim__(key, item)))
                elif isinstance(item, SparseTensor):
                    for k in ptrs[key].keys():
                        value = getattr(item.storage, k)
                        store[key][k].append(value)
                        ptrs[key][k].append(ptrs[key][k][-1] + value.size(0))

                    store[key]['__data_class__'] = item.__class__
                    store[key]['_sparse_sizes'].append(
                        item.storage._sparse_sizes)
                elif isinstance(item, list) or isinstance(item, tuple):
                    store[key].append(item)
                    ptrs[key].append(ptrs[key][-1] + len(item))
                else:
                    store[key].append(item)
                    ptrs[key].append(ptrs[key][-1] + 1)

        for key in keys:
            item = ref[key]
            if torch.is_tensor(item):
                store[key] = torch.cat(store[key],
                                       dim=ref.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                for k in ptrs[key].keys():
                    store[key][k] = torch.cat(store[key][k])

        assert '__data_class__' not in store
        store['__data_class__'] = ref.__class__
        assert '__length__' not in store
        store['__length__'] = len(data_list)

        return store, ptrs
