from math import pi as PI
import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.matlab import read_matlab


class MNISTSuperpixel75(Dataset):

    url = "http://geometricdeeplearning.com/code/MoNet/MoNet_code.tar.gz"

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(MNISTSuperpixel75, self).__init__()

        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.data_file)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))
        dir = os.path.join(self.raw_folder, 'MoNet_code', 'MNIST', 'datasets')
        read_mnist_monet(dir)
        # torch.save(data, self.data_file)

        print('Done!')


def read_mnist_monet(dir):
    data_dir = os.path.join(dir, 'mnist_superpixels_data_75')

    test_centroids = read_matlab(
        os.path.join(data_dir, 'test_centroids.mat'), 'idx_centroids')
    print('test_centroids', test_centroids.size())

    train_centroids = read_matlab(
        os.path.join(data_dir, 'train_centroids.mat'), 'idx_centroids')
    print('train_centroids', train_centroids.size())

    test_patch_coords = read_matlab(
        os.path.join(data_dir, 'test_patch_coords.mat'), 'patch_coords')
    print('test_patch_coords', test_patch_coords.size())

    # train_patch_coords = read_matlab(
    #     os.path.join(data_dir, 'train_patch_coords.mat'), 'patch_coords')
    # print('train_patch_coords', train_patch_coords.size())

    test_vals = read_matlab(os.path.join(data_dir, 'test_vals.mat'), 'vals')
    print('test_vals', test_vals.shape)

    train_vals = read_matlab(os.path.join(data_dir, 'train_vals.mat'), 'vals')
    print('train_vals', train_vals.shape)

    test_labels = read_matlab(
        os.path.join(dir, 'MNIST_preproc_test_labels', 'MNIST_labels.mat'),
        'labels')
    print('test_labels', test_labels.size())

    train_labels = read_matlab(
        os.path.join(dir, 'MNIST_preproc_train_labels', 'MNIST_labels.mat'),
        'labels')
    print('train_labels', train_labels.size())

    test_rho = test_patch_coords[:, :75 * 75].contiguous().view(-1, 75, 75)
    test_theta = test_patch_coords[:, 75 * 75:].contiguous().view(-1, 75, 75)
    test_rho[test_rho != test_rho] = 0  # Set NaNs to zero.
    test_theta += PI
    test_theta[test_theta != test_theta] = 0  # Set NaNs to zero.

    # print(test_rho.size())
    # print(test_theta.size())
    # print(test_rho[0, :5, :5])
    # print(test_rho.max())
    # print(test_theta[0, :5, :5])
    # print(test_theta.max())
    # print(test_theta.min())

    print(test_rho[0].size())
    test_index = [test_rho[i].nonzero() for i in range(10000)]
    print(test_index[0].size())
    print(test_index[1].size())
    # test_rho[0][test_index[0].t()]
    # test_rho = [test_rho[i][test_index[i]] for i in range(1000)]
    print(len(test_index))
    # print(test_index[0][:15])


# from __future__ import division

# import os
# import h5py
# import numpy as np

# path_train_vals = os.path.join('./mnist/train_vals.mat')
# path_test_vals = os.path.join('./mnist/test_vals.mat')
# path_train_coords = os.path.join('./mnist/train_patch_coords.mat')
# path_test_coords = os.path.join('./mnist/test_patch_coords.mat')
# path_train_centroids = os.path.join('./mnist/train_centroids.mat')
# path_test_centroids = os.path.join('./mnist/test_centroids.mat')
# path_train_labels = os.path.join('./mnist/train_labels.mat')
# path_test_labels = os.path.join('./mnist/test_labels.mat')

# def load_matlab_file(path_file, name_field, dtype=np.float32):
#     db = h5py.File(path_file, 'r')
#     ds = db[name_field]
#     out = np.asarray(ds).astype(dtype).T.squeeze()
#     db.close()
#     return out

# train_vals = load_matlab_file(path_train_vals, 'vals')
# test_vals = load_matlab_file(path_test_vals, 'vals')
# train_coords = load_matlab_file(path_train_coords, 'patch_coords')
# test_coords = load_matlab_file(path_test_coords, 'patch_coords')
# train_centroids = load_matlab_file(path_train_centroids, 'idx_centroids',
#                                    np.int64)
# test_centroids = load_matlab_file(path_test_centroids, 'idx_centroids',
#                                   np.int64)
# print(test_centroids)
# train_labels = load_matlab_file(path_train_labels, 'labels', np.uint8)
# test_labels = load_matlab_file(path_test_labels, 'labels', np.uint8)

# # 75 * 75 * 2 values (warum 2?)
# # 1-dim centroids? wtf?
# # dist is 0 at root, rad is NaN

# # I don't really need this if I have the centroids. I just need the row and col
# # array
# # sigma needs to be calculated, not passed
# # TODO: find a way to extract these arrays
# def stack_matrices(x):  # Stack 2D matrices into a 3D tensor
#     n = int(np.sqrt(x.shape[1] // 2))
#     num_examples = x.shape[0]

#     rho = x[:, :n * n]
#     theta = x[:, n * n:]
#     rho = np.reshape(rho, (num_examples, n, n, 1))
#     theta = np.reshape(theta, (num_examples, n, n, 1))
#     return np.concatenate([rho, theta], axis=3)

# # Load the coords
# test_coords = stack_matrices(test_coords)
# train_coords = stack_matrices(train_coords)

# # Compute centroids
# def compute_centroids(centroids, list_mapping_graph_0_to_graph_1,
#                       list_mapping_graph_1_to_graph_2):
#     pass
