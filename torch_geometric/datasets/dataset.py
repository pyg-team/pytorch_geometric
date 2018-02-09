import os.path as osp

import torch
from torch.utils.data import Dataset as BaseDataset

from ..sparse import SparseTensor
from .utils.dir import make_dirs


def _to_list(x):
    return x if isinstance(x, list) else [x]


def _exists(files):
    return all([osp.exists(f) for f in _to_list(files)])


def _data_list_to_set(data_list):
    size = len(data_list)

    input = [None] * size
    pos = [None] * size
    index = [None] * size
    weight = [None] * size
    target = [None] * size

    slice, index_slice = [0], [0]

    for i, data in enumerate(data_list):
        input.append(data.input)
        pos.append(data.pos)
        index.append(data.index)
        weight.append(data.weight)
        target.append(data.target)

        slice.append(slice[-1] + data.num_nodes)
        index_slice.append(index_slice[-1] + data.num_edges)

    input = None if input[0] is None else torch.cat(input, dim=0)
    pos = None if pos[0] is None else torch.cat(pos, dim=0)
    index = None if index[0] is None else torch.cat(index, dim=1)
    weight = None if weight[0] is None else torch.cat(weight, dim=0)
    target = None if target[0] is None else torch.cat(target, dim=0)

    slice = torch.LongTensor(slice)
    index_slice = torch.LongTensor(index_slice)

    return _Set(input, pos, index, weight, target, slice, index_slice)


def data_list_to_batch(data_list):
    raise NotImplementedError


def _cat(data_list, key):
    if getattr(data_list[0], key, None) is None:
        return None


class Data(object):
    def __init__(self, input, pos, index, weight, target):
        self.input = input
        self.pos = pos
        self.index = index
        self.weight = weight
        self.target = target
        self.batch = None

    @property
    def _props(self):
        props = ['input', 'pos', 'index', 'weight', 'target', 'batch']
        return [p for p in props if getattr(self, p, None) is not None]

    @property
    def adj(self):
        n = self.num_nodes
        size = torch.Size([n, n] + self.weight.size()[1:])
        return SparseTensor(self.index, self.weight, size)

    @property
    def num_nodes(self):
        assert self.input is not None or self.pos is not None, (
            'At least input or position tensor must be defined in order to '
            'compute number of nodes')

        if self.pos is None:
            return self.input.size(0)
        else:
            return self.pos.size(0)

    @property
    def num_edges(self):
        assert self.index is not None or self.weight is not None, (
            'At least index or weight tensor must be defined in order to '
            'compute number of edges')

        if self.weight is None:
            return self.index.size(1)
        else:
            return self.weight.size(0)

    def _transer(self, device, props=None):
        props = self._props if props is None else _to_list(props)
        for prop in props:
            getattr(getattr(self, prop), device)()

    def cuda(self, props=None):
        self._transer('cuda', props)

    def cpu(self, props=None):
        self._transer('cpu', props)


class _Set(Data):
    def __init__(self, input, pos, index, weight, target, slice, index_slice):
        super(Data, self).__init__(input, pos, index, weight, target)
        self.slice = slice
        self.index_slice = index_slice

    def __getitem__(self, i):
        s1, s2 = self.slice, self.index_slice
        input = None if self.input is None else self.input[s1[i]:s1[i + 1]]
        pos = None if self.pos is None else self.pos[s1[i]:s1[i + 1]]
        index = None if self.index is None else self.index[:, s2[i]:s2[i + 1]]
        weight = None if self.weight is None else self.weight[s2[i]:s2[i + 1]]

        target = self.target
        if target is not None and self.num_nodes == target.size(0):
            target = target[s1[i]:s1[i + 1]]
        else:
            target = target[i]

        return Data(input, pos, index, weight, target)

    def __len__(self):
        return self.slice.size(0) - 1


class Dataset(BaseDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__()

        self.root = osp.expanduser(osp.normpath(root))
        self.raw_folder = osp.join(self.root, 'raw')
        self.processed_folder = osp.join(self.root, 'processed')
        self.transform = transform

        self._download()
        self._process()

    @property
    def raw_files(self):
        raise NotImplementedError

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    @property
    def _raw_files(self):
        files = _to_list(self.raw_files)
        return [osp.join(self.raw_folder, f) for f in files]

    @property
    def _processed_files(self):
        files = _to_list(self.processed_files)
        return [osp.join(self.processed_folder, f) for f in files]

    def __getitem__(self, i):
        data = self.set[i]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.set)

    def _download(self):
        if _exists(self._raw_files):
            return

        self.download()

    def _process(self):
        if _exists(self._processed_files):
            return

        make_dirs(self.processed_folder)
        sets = self.process()
        sets = sets if isinstance(sets, tuple) else (sets, )

        # Save (training and test) sets separately.
        for i in range(len(self._processed_files)):
            torch.save(_data_list_to_set(sets[i]), self._processed_files[i])
