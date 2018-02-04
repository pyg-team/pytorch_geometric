import os.path as osp
import collections

import torch
from torch.utils.data import Dataset as BaseDataset

from ..sparse import SparseTensor


def to_list(x):
    return x if isinstance(x, collections.Sequence) else [x]


def exists(files):
    return all([osp.exists(f) for f in to_list(files)])


class Data(object):
    def __init__(self, input, position, index, weight, target):
        self.input = input
        self.position = position
        self.index = index
        self.weight = weight
        self.target = target

    @property
    def adj(self):
        n = self.input.size(0)
        size = torch.Size([n, n] + self.weight.size()[1:])
        return SparseTensor(self.index, self.weight, size)


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
        files = to_list(self.raw_files)
        return [osp.join(self.raw_folder, f) for f in files]

    @property
    def _processed_files(self):
        files = to_list(self.processed_files)
        return [osp.join(self.processed_folder, f) for f in files]

    def __getitem__(self, i):
        data = self.datas[i]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.datas)

    def _download(self):
        if exists(self._raw_files):
            return

        self.download()

    def _process(self):
        if exists(self._processed_files):
            return

        sets = self.process()
        sets = sets if isinstance(sets, tuple) else (sets, )

        # Save training and test set separately.
        for i in range(len(self._processed_files)):
            torch.save(to_list(sets[i]), self._processed_files[i])
