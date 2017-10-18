import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..graph.geometry import edges_from_faces, mesh_adj
from .utils.dir import make_dirs
from .utils.ply import read_ply
from .utils.save import save


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
    n_training = 80
    n_test = 20

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        self.root = os.path.expanduser(root)
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.process()

        # data_file = self.training_file if train else self.test_file
        # path = os.path.join(self.root, self.processed_folder, data_file)
        # self.data, self.labels = torch.load(path)

    def __len__(self):
        return self.n_training if self.train else self.n_test

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.training_file) and \
               os.path.exists(self.test_file)

    def _read_example(self, index):
        path = os.path.join(self.root, 'training', 'registrations',
                            'tr_reg_{0:03d}.ply'.format(index))

        vertices, faces = read_ply(path)
        edges = edges_from_faces(faces)

        return vertices, edges

    def _save_examples(self, indices, path):
        data = [self._read_example(i) for i in indices]

        vertices = torch.stack([example[0] for example in data], dim=0)
        edges = torch.stack([example[1] for example in data], dim=0)

        save({'vertices': vertices, 'edges': edges}, path)

    def process(self):
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))

        train_indices = range(0, self.n_training)
        self._save_examples(train_indices, self.training_file)

        test_indices = range(self.n_training, self.n_training + self.n_test)
        self._save_examples(test_indices, self.test_file)

        print('Done!')


# p1 = os.path.expanduser('~/Downloads/MPI-FAUST/training/scans/tr_scan_000.ply')
# p2 = os.path.expanduser(
#     '~/Downloads/MPI-FAUST/training/registrations/tr_reg_001.ply')
# p3 = os.path.expanduser(
#     '~/Downloads/MPI-FAUST/training/ground_truth_vertices/tr_gt_000.txt')

# read_ply(p1)
# read_ply(p2)

# def read_with_numpy(path):
#     with open(path, 'rb') as f:

#         print(f)

# # TODO:  COmpute 544-dimensional SHOT descriptors (local histogram of normal
# # vectors)
# # Architecture:
# # * Lin16
# # * ReLU
# # * CNN32
# # * AMP
# # * ReLU
# # * CNN64
# # * AMP
# # * ReLU
# # * CNN128
# # * AMP
# # * ReLU
# # * Lin256
# # * Lin6890
# #
# # output representing the soft correspondence as an 6890-dimensional vector.
# #
# # shape correspondence (labelling problem) where one tries to label each vertex
# # to the index of a corresponding point.
# # X = Indices
# # y_j denote the vertex corresponding to x_i
# # output representing the probability distribution on Y
# # loss: multinominal regression loss
# # ACN: adam optimizer 10^-3, beta_1 = 0.9, beta_2 = 0.999

# # SHOT descriptor: Salti, Tombari, Stefano. SHOT, 2014
# # Signature of Histograms of OrienTations (SHOT)

# # v, f = read_ply(p1)
# # print(v.shape)
# # print(f.shape)
# # # TODO: really slow
# # read_with_numpy(p1)
# read_ply(p2)

# with open(p3, 'rb') as f:
#     content = f.readlines()
#     print(len(content))

# read_with_numpy(p2)
