import os

from plyfile import PlyData, make2d
import numpy as np
import torch
from torch.utils.data import Dataset


class FAUST(Dataset):
    """`MPI FAUST <http://faust.is.tue.mpg.de/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where
            ``processed/training.pt`` and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    url = 'http://faust.is.tue.mpg.de/'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.process()

        data_file = self.training_file if train else self.test_file
        path = os.path.join(self.root, self.processed_folder, data_file)
        self.data, self.labels = torch.load(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((adj_indices, points, features), target) where target is
            index of the target class.
        """

        data, target = self.data[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.labels)

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file))

    def process(self):
        if self._check_processed():
            return

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

        # Process and save as torch files.
        print('Processing...')

        print('Done!')


# FAUST('~/Downloads/MPI-FAUST')

p1 = os.path.expanduser('~/Downloads/MPI-FAUST/training/scans/tr_scan_000.ply')


def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        vertices = np.stack((x, y, z), axis=1)
        faces = make2d(plydata['face']['vertex_indices'])
        return vertices, faces


v, f = read_ply(p1)
print(v.shape)
print(f.shape)
# TODO: really slow
