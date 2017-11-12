import os
from math import pi as PI

import torch
from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar


class FAUSTPatch(Dataset):

    shot_url = 'http://www.roemisch-drei.de/faust_shot.tar.gz'
    n_training = 80
    n_test = 20

    def __init__(self,
                 root,
                 train=True,
                 shot=False,
                 correspondence=False,
                 transform=None,
                 target_transform=None):

        super(FAUSTPatch, self).__init__()

        self.root = os.path.expanduser(root)
        self.shot_folder = os.path.join(self.root, 'shot')
        self.processed_folder = os.path.join(self.root, 'processed', 'patch')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.correspondence = correspondence
        self.transform = transform
        self.target_transform = target_transform

        self.download()

        data_file = self.training_file if train else self.test_file
        self.index, self.weight, self.slice = torch.load(data_file)

        if shot is True:
            data_file = os.path.join(
                self.shot_folder,
                'training_shot.pt') if train else os.path.join(
                    self.shot_folder, 'test_shot.pt')
            self.shot = torch.load(data_file)
        else:
            self.shot = None

    def __getitem__(self, i):
        start, end = self.slice[i:i + 2]
        index = self.index[:, start:end]
        weight = self.weight[start:end]
        weight = to_polar(weight)
        n = 6890
        if self.shot is None:
            input = torch.FloatTensor(n, 1).fill_(1)
        else:
            input = self.shot[i]
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n, 2]))
        data = (input, adj, torch.FloatTensor())

        if self.correspondence:
            target = torch.arange(0, input.size(0)).long()
        else:
            target = index % 10  # Every subject is representated by 10 poses.

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.correspondence and not self.train:
            geodesic_distances = torch.load(
                os.path.join(self.processed_folder, '../geodesic_distances',
                             '{:02d}.pt'.format(i)))
            return data, target, geodesic_distances
        else:
            return data, target

    def __len__(self):
        return self.n_training if self.train else self.n_test

    def _check_exists(self):
        return os.path.exists(self.root)

    def _save_examples(self, indices, path):
        data = [self._read_example(i) for i in indices]

        vertices = torch.stack([example[0] for example in data], dim=0)
        edges = torch.stack([example[1] for example in data], dim=0)

        torch.save((vertices, edges), path)

    def download(self):
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

        if not os.path.exists(self.shot_folder):
            print('Downloading {}'.format(self.shot_url))

            file_path = download_url(self.shot_url, self.shot_folder)
            extract_tar(file_path, self.shot_folder)
            os.unlink(file_path)


def to_polar(weight):
    scale = torch.FloatTensor([weight[:, 0].max(), 2 * PI])
    weight /= scale
    weight += torch.FloatTensor([0, 0.5])
    return weight


def to_cartesian(weight):
    x = weight[:, 0] * torch.cos(weight[:, 1])
    y = weight[:, 0] * torch.sin(weight[:, 1])
    weight = torch.stack([x, y], dim=1)
    c = 1 / (2 * weight.abs().max())
    weight = c * weight + 0.5
    return weight
