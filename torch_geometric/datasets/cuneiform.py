import os

import torch
from torch.utils.data import Dataset

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
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform

        self.download()
        self.process()

        # Load processed data.

    def __getitem__(self, i):
        pass

    def __len__(self):
        pass

    @property
    def _raw_exists(self):
        files = ['{}_{}.txt'.format(self.prefix, f) for f in self.filenames]
        files = [os.path.join(self.raw_folder, f) for f in files]
        return all([os.path.exists(f) for f in files])

    @property
    def _processed_exists(self):
        train_exists = os.path.exists(self.training_file)
        test_exists = os.path.exists(self.test_file)
        return train_exists and test_exists

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        if self._processed_exists:
            return

        # spinner = Spinner('Processing').start()
        make_dirs(self.processed_folder)

        dir = self.raw_folder
        index = read_file(dir, self.prefix, 'A').long()
        slice = read_file(dir, self.prefix, 'graph_indicator').long()
        target = read_file(dir, self.prefix, 'graph_labels').byte()
        position = read_file(dir, self.prefix, 'node_attributes')
        input = read_file(dir, self.prefix, 'node_labels').byte()
        print(index.size())
        print(slice.size())
        print(target.size())
        print(position.size())
        print(input.size())

        # spinner.success()
