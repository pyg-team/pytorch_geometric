import os

import torch
from torch.utils.data import Dataset

from .data import Data
from ..sparse import SparseTensor
from .utils.download import download_url
from .utils.extract import extract_tar


class MNISTSuperpixels(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist>`_ Superpixels Dataset from the
    `"Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper. Each example contains exactly
    75 graph-nodes.

    Args:
        root (string): Root directory of dataset. Download data automatically
            if dataset doesn't exist.
        train (bool, optional): If ``True``, creates dataset from
            ``training.pt``, otherwise from ``test.pt``. (default: ``True``)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    url = "http://www.roemisch-drei.de/mnist_superpixels.tar.gz"

    def __init__(self, root, train=True, transform=None):
        super(MNISTSuperpixels, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.training_file = os.path.join(self.raw_folder, 'training.pt')
        self.test_file = os.path.join(self.raw_folder, 'test.pt')

        self.train = train
        self.transform = transform

        # Download data.
        self.download()

        # Load processed data.
        data_file = self.training_file if train else self.test_file
        input, index, slice, position, target = torch.load(data_file)
        self.input, self.index, self.slice = input, index.long(), slice
        self.position, self.target = position, target.long()

    def __getitem__(self, i):
        input = self.input[i]
        index = self.index[:, self.slice[i]:self.slice[i + 1]]
        weight = torch.ones(index.size(1))
        position = self.position[i]
        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        target = self.target[i]
        data = Data(input, adj, position, target)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.input.size(0)

    def _raw_exists(self):
        return os.path.exists(self.raw_folder)

    def download(self):
        if self._raw_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)
