import os

import torch
from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar


class MNISTSuperpixel75(Dataset):

    url = "http://www.roemisch-drei.de/mnist_superpixel_75.tar.gz"

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(MNISTSuperpixel75, self).__init__()

        self.root = os.path.expanduser(root)
        self.training_file = os.path.join(self.root, 'training.pt')
        self.test_file = os.path.join(self.root, 'test.pt')

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download()

        data_file = self.training_file if train else self.test_file
        input, index, slice, position, target = torch.load(data_file)
        self.input, self.index, self.slice = input, index, slice
        self.position, self.target = position, target

    def __getitem__(self, i):
        input = self.input[i]
        position = self.position[i]
        index = self.index[:, self.slice[i]:self.slice[i + 1]].long()
        weight = input.new(index.size(1)).fill_(1)
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([75, 75]))
        data = (input, adj, position)
        target = self.target[i]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.input.size(0)

    def _check_exists(self):
        return os.path.exists(self.root)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.root)
        extract_tar(file_path, self.root)
        os.unlink(file_path)
