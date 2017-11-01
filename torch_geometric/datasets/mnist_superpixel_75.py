import os

import torch
from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar


class MNISTSuperpixel75(Dataset):

    url = "http://www.roemisch-drei.de/mnist_superpixel_75.tar.gz"
    n_training = 60000
    n_test = 10000

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(MNISTSuperpixel75, self).__init__()

        self.root = os.path.expanduser(root)
        self.data_file = os.path.join(self.root, 'data.pt')

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download()

        input, position, index, slice, target = torch.load(self.data_file)
        self.input, self.position = input, position
        self.index, self.slice, self.target = index, slice, target

    def __getitem__(self, i):
        input = self.input[i]
        position = self.position[i]
        index = self.index[:, self.slice[i]:self.slice[i + 1]].long()
        weight = input.new(index.size(1)).fill_(1)
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([75, 75]))
        data = (input, position, adj)
        target = self.target[i]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.n_training if self.train else self.n_test

    def _check_exists(self):
        return os.path.exists(self.root)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.root)
        extract_tar(file_path, self.root)
        os.unlink(file_path)
