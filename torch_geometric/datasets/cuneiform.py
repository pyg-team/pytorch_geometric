from __future__ import division

import os

import torch
from torch.utils.data import Dataset

from .data import Data
from ..sparse import SparseTensor
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.dir import make_dirs
from .utils.spinner import Spinner
from .utils.tu_format import read_file, read_adj, read_slice


class Cuneiform(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    prefix = 'CuneiformArrangement'
    filenames = [
        'A', 'graph_indicator', 'graph_labels', 'node_labels',
        'node_attributes'
    ]

    def __init__(self, root, split=None, transform=None):
        super(Cuneiform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.transform = transform

        self.download()
        self.process()

        # Load processed data.
        data = torch.load(self.data_file)
        input, index, position, target, slice, index_slice = data
        self.input, self.index, self.position = input, index, position
        self.target, self.slice, self.index_slice = target, slice, index_slice

        if split is not None:
            self.split = split
        else:
            self.split = torch.arange(
                0, target.size(0), out=torch.LongTensor())

    def __getitem__(self, i):
        i = self.split[i]
        input = self.input[self.slice[i]:self.slice[i + 1]]
        index = self.index[:, self.index_slice[i]:self.index_slice[i + 1]]
        weight = input.new(index.size(1)).fill_(1)
        position = self.position[self.slice[i]:self.slice[i + 1]]
        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        target = self.target[i]
        data = Data(input, adj, position, target)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.split.size(0)

    @property
    def _raw_exists(self):
        files = ['{}_{}.txt'.format(self.prefix, f) for f in self.filenames]
        files = [os.path.join(self.raw_folder, f) for f in files]
        return all([os.path.exists(f) for f in files])

    @property
    def _processed_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        if self._processed_exists:
            return

        spinner = Spinner('Processing').start()
        make_dirs(self.processed_folder)

        index, index_slice = read_adj(self.raw_folder, self.prefix)
        slice = read_slice(self.raw_folder, self.prefix)
        position = read_file(self.raw_folder, self.prefix, 'node_attributes')
        depth = position[:, 2]
        depth = (depth - depth.mean()) / max(depth.std(), 1.0 / depth.size(0))
        depth = depth.view(-1, 1)
        position = position[:, :2]
        target = read_file(self.raw_folder, self.prefix, 'graph_labels').long()

        # Encode inputs.
        x = read_file(self.raw_folder, self.prefix, 'node_labels')
        x += torch.FloatTensor([0, 4])
        x += torch.arange(0, 7 * x.size(0), 7).view(-1, 1)
        input = torch.zeros(7 * x.size(0))
        input[x.view(-1).long()] = 1
        input = input.view(-1, 7)
        input = torch.cat([input, depth], dim=1)

        data = (input, index, position, target, slice, index_slice)
        torch.save(data, self.data_file)

        spinner.success()
