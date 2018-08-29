import collections
import os.path as osp

import torch.utils.data

from .makedirs import makedirs


def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    return all([osp.exists(f) for f in files])


class Dataset(torch.utils.data.Dataset):
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

    def __len__(self):
        raise NotImplementedError

    def get(self, idx):
        raise NotImplementedError

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(Dataset, self).__init__()

        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self._download()
        self._process()

    @property
    def raw_paths(self):
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):
            return

        makedirs(self.raw_dir)
        self.download()

    def _process(self):
        if files_exist(self.processed_paths):
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        print('Done!')

    def __getitem__(self, idx):
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
