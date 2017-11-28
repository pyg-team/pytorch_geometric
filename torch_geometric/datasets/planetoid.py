import os

import torch
from torch.utils.data import Dataset

from .data import Data
from ..sparse import SparseTensor
from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.planetoid import read_planetoid
from .utils.spinner import Spinner


class Planetoid(Dataset):
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    ext = ['tx', 'ty', 'allx', 'ally', 'graph', 'test.index']

    def __init__(self, root, name, transform=None):
        super(Planetoid, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.name = name
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')
        self.transform = transform

        # Download and process data.
        self.download()
        self.process()

        # Load processed data.
        data = torch.load(self.data_file)
        input, index, target = data

        # Create unweighted sparse adjacency matrix.
        weight = torch.ones(index.size(1))
        n = input.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))

        # Bundle graph to data object.
        self.data = Data(input, adj, position=None, target=target.long())

    def __getitem__(self, index):
        data = self.data

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return 1

    @property
    def _raw_exists(self):
        files = ['ind.{}.{}'.format(self.name, ext) for ext in self.ext]
        files = [os.path.join(self.raw_folder, f) for f in files]
        files = [os.path.exists(f) for f in files]
        return all(files)

    @property
    def _processed_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, 'data.pt'))

    def download(self):
        if self._raw_exists:
            return

        for ext in self.ext:
            url = '{}/ind.{}.{}'.format(self.url, self.name, ext)
            download_url(url, self.raw_folder)

    def process(self):
        if self._processed_exists:
            return

        spinner = Spinner('Processing').start()
        make_dirs(os.path.join(self.processed_folder))
        data = read_planetoid(self.raw_folder, self.name)
        torch.save(data, self.data_file)
        spinner.success()
