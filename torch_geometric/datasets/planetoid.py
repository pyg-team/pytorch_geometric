from __future__ import print_function

import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.planetoid import read_planetoid


class Planetoid(Dataset):
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(self, root, name, transform=None, target_transform=None):
        super(Planetoid, self).__init__()

        self.root = os.path.expanduser(root)
        self.name = name
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

        # Load processed data.
        data = torch.load(self.data_file)
        self.input, index, self.target, self.test_mask = data

        # Create unweighted sparse adjacency matrix.
        weight = torch.ones(index.size(1))
        n = self.input.size(0)
        self.adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

        # Generate train and validation mask.
        self.train_mask = torch.arange(0, 20 * (self.target.max() + 1)).long()
        len_y = self.train_mask.size(0)
        self.val_mask = torch.arange(len_y, len_y + 500).long()

    def __getitem__(self, index):
        data = (self.input, self.adj, None)
        target = self.target

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (data, target)

    def __len__(self):
        return 1

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.data_file)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        ext = ['tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
        for e in ext:
            url = '{}/ind.{}.{}'.format(self.url, self.name, e)
            download_url(url, self.raw_folder)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))
        data = read_planetoid(self.raw_folder, self.name)
        torch.save(data, self.data_file)

        print('Done!')
