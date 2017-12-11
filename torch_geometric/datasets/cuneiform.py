import os

import torch
from torch.utils.data import Dataset
import numpy as np

from .data import Data
from ..sparse import SparseTensor
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.dir import make_dirs
from .utils.spinner import Spinner
from .utils.tu_format import read_file


class Cuneiform(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    prefix = 'CuneiformArrangement'
    filenames = [
        'A', 'graph_indicator', 'graph_labels', 'node_labels',
        'node_attributes'
    ]

    def __init__(self, root, train=True, transform=None):
        super(Cuneiform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.train = train
        self.transform = transform

        self.download()
        self.process()

        # Load processed data.
        data = torch.load(self.data_file)
        input, index, position, target, slice, index_slice = data
        self.input, self.index = input.float(), index.long()
        self.position, self.target = position, target.long()
        self.slice, self.index_slice = slice, index_slice
        self.n = self.target.size(0)

    def __getitem__(self, i):
        i = i if self.train else i + self.n - 60
        input = self.input[self.slice[i]:self.slice[i + 1]]
        index = self.index[:, self.index_slice[i]:self.index_slice[i + 1]]
        weight = torch.ones(index.size(1))
        position = self.position[self.slice[i]:self.slice[i + 1]]
        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        target = self.target[i]
        data = Data(input, adj, position, target)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.n - 60 if self.train else 60

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

        dir = self.raw_folder
        index = read_file(dir, self.prefix, 'A').long()
        slice = read_file(dir, self.prefix, 'graph_indicator').view(-1).long()
        target = read_file(dir, self.prefix, 'graph_labels').view(-1).byte()
        position = read_file(dir, self.prefix, 'node_attributes')
        input = read_file(dir, self.prefix, 'node_labels').byte()

        # Convert to slice representation.
        slice = np.bincount(slice.numpy())
        for i in range(1, len(slice)):
            slice[i] = slice[i - 1] + slice[i]
        slice = torch.LongTensor(slice)

        # Convert to none-one-hot vector.
        x = input.new(input.size(0) * 7).fill_(0)
        input += torch.ByteTensor([0, 4])
        y = torch.arange(0, input.size(0)).view(-1, 1).long() * 7
        input = input.long() + y
        x[input.view(-1)] = 1
        input = x.view(-1, 7)

        index_slice = [0]
        index -= 1
        for i in range(index.size(0)):
            curr_idx = len(index_slice)
            row, col = index[i]
            if row >= slice[curr_idx]:
                index_slice.append(i)
                curr_idx += 1

            index[i, :] -= slice[curr_idx - 1]

        index_slice.append(index.size(0))
        index_slice = torch.LongTensor(index_slice)
        index = index.byte().t()

        data = (input, index, position, target, slice, index_slice)
        torch.save(data, self.data_file)

        spinner.success()
