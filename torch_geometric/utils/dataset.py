import collections
import os.path as osp

import torch.utils.data


def to_list(x):
    return x if isinstance(x, collections.Iterable) else [x]


def files_exist(files):
    return all([osp.exists(f) for f in files])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__()

        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform

        self._download()
        self._process()

    @property
    def raw_files(self):
        raise NotImplementedError

    @property
    def processed_files(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_data(self, i):
        raise NotImplementedError

    @property
    def _raw_files(self):
        files = to_list(self.raw_files)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def _processed_files(self):
        files = to_list(self.processed_files)
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self._raw_files):
            return

        self.download()

    def _process(self):
        if files_exist(self._processed_files):
            return

        self.process()

    def __getitem__(self, i):
        data = self.get_data(i)

        if self.transform is not None:
            data = self.transform(data)

        return data
