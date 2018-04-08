import os

import torch

from .dataset import Dataset, Set
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.spinner import Spinner


class MNISTSuperpixels2(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist>`_ Superpixels Dataset from the
    `"Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper. Each example contains exactly
    75 graph-nodes.

    Args:
        root (string): Root directory of dataset. Downloads data automatically
            if dataset doesn't exist.
        train (bool, optional): If :obj:`True`, creates dataset from
            ``training.pt``, otherwise from ``test.pt``. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
          'mnist_superpixels.tar.gz'

    def __init__(self, root, train=True, transform=None):
        super(MNISTSuperpixels2, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        spinner = Spinner('Processing').start()

        sets = ()

        for file_name in self._raw_files:
            input, index, index_slice, pos, target = torch.load(file_name)
            g, n = input.size()
            slice = torch.arange(0, (1 + g) * n, n, out=torch.LongTensor())
            input, pos = input.view(-1), pos.view(-1, 2)
            index, target = index.long(), target.long()

            set = Set(input, pos, index, None, target, None, slice,
                      index_slice)
            sets += (set, )

        spinner.success()

        return sets
