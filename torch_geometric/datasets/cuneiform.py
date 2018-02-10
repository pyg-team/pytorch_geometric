from __future__ import division

import os

import torch

from .dataset import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.spinner import Spinner


class Cuneiform(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    prefix = 'CuneiformArrangement'
    filenames = [
        'A', 'graph_indicator', 'graph_labels', 'node_labels',
        'node_attributes'
    ]

    def __init__(self, root, split=None, transform=None):
        super(Cuneiform, self).__init__(root, transform)
        self.set = torch.load(self._processed_folder[0])

        if split is None:
            self.split = torch.arange(0, self.set.num_nodes).long()
        else:
            self.split = split

    def __getitem__(self, i):
        super[self.split[i]]

    def __len__(self):
        return self.split.size(0)

    @property
    def raw_files(self):
        return ['{}_{}.txt'.format(self.prefix, f) for f in self.filenames]

    @property
    def processed_files(self):
        return 'data.pt'

    def download(self):
        print('hello')
        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        spinner = Spinner('Processing').start()
        spinner.success()
