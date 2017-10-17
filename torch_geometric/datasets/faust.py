import os

from plyfile import PlyData, make2d
import numpy as np
import torch
from torch.utils.data import Dataset

from ..graph.geometry import adj_indices_from_faces


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

        # data_file = self.training_file if train else self.test_file
        # path = os.path.join(self.root, self.processed_folder, data_file)
        # self.data, self.labels = torch.load(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((adj_indices, points, features), target) where target is
            index of the target class.
        """

        return 1, 2
        # data, target = self.data[index], self.labels[index]

        # if self.transform is not None:
        #     data = self.transform(data)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return data, target

    def __len__(self):
        pass

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file))

    def _read_ply(self, index):
        path = os.path.join(self.root, 'training', 'registrations',
                            'tr_reg_{0:03d}.ply'.format(index))

        with open(path, 'rb') as f:
            plydata = PlyData.read(f)

        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        vertices = np.stack((x, y, z), axis=1)
        faces = make2d(plydata['face']['vertex_indices'])

        vertices = torch.FloatTensor(vertices)
        faces = torch.IntTensor(faces)

        adj_indices = adj_indices_from_faces(faces)

        return vertices, adj_indices

        print(vertices.shape)
        print(faces.shape)
        # return vertices, faces

    def process(self):
        if self._check_processed():
            return

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

        # Process and save as torch files.
        print('Processing...')

        train_indices = range(0, 2)
        test_indices = range(80, 20)

        for i in train_indices:
            self._read_ply(i)

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
