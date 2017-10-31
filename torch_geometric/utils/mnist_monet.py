from math import pi as PI
import os

import h5py
import torch
import numpy as np


def read_matlab(path, name):
    db = h5py.File(path, 'r')
    ds = db[name]
    out = np.asarray(ds).astype(np.float32).T.squeeze()
    db.close()
    return torch.FloatTensor(out)


def read_mnist_monet(dir):
    data_dir = os.path.join(dir, 'mnist_superpixels_data_75')

    # Get the input.
    train_input = read_matlab(os.path.join(data_dir, 'train_vals.mat'), 'vals')
    test_input = read_matlab(os.path.join(data_dir, 'test_vals.mat'), 'vals')
    input = torch.cat([train_input, test_input], dim=0)

    # Get the target.
    train_target = read_matlab(
        os.path.join(dir, 'MNIST_preproc_train_labels', 'MNIST_labels.mat'),
        'labels')
    test_target = read_matlab(
        os.path.join(dir, 'MNIST_preproc_test_labels', 'MNIST_labels.mat'),
        'labels')
    target = torch.cat([train_target, test_target], dim=0)

    # Get the polar coordinate adjacency.
    train_patch_coord = read_matlab(
        os.path.join(data_dir, 'train_patch_coords.mat'), 'patch_coords')
    train_rho = train_patch_coord[:, :75 * 75].contiguous().view(-1, 75, 75)
    train_theta = train_patch_coord[:, 75 * 75:].contiguous().view(-1, 75, 75)
    train_rho[train_rho != train_rho] = 0
    train_theta[train_theta != train_theta] = 0
    test_patch_coord = read_matlab(
        os.path.join(data_dir, 'test_patch_coords.mat'), 'patch_coords')
    test_rho = test_patch_coord[:, :75 * 75].contiguous().view(-1, 75, 75)
    test_theta = test_patch_coord[:, 75 * 75:].contiguous().view(-1, 75, 75)
    test_rho[test_rho != test_rho] = 0
    test_theta[test_theta != test_theta] = 0
    rho = torch.cat([train_rho, test_rho], dim=0)
    theta = torch.cat([train_theta, test_theta], dim=0)

    return input, target, rho, theta


def polar2euclidean(rho, theta):
    nz = rho.nonzero().t()
    rho = rho[nz[0], nz[1]]
    theta = theta[nz[0], nz[1]]

    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    euclidean = torch.stack([x, y], dim=1)

    return torch.sparse.FloatTensor(nz, euclidean, torch.Size([75, 75, 2]))


path = os.path.expanduser('~/Downloads/MoNet_code/MNIST')
# data = read_mnist_monet(os.path.join(path, 'datasets'))
# torch.save(data, os.path.join(path, 'data.pt'))
data = torch.load(os.path.join(path, 'data.pt'))
input, target, rho, theta = data

for i in range(2):
    a = polar2euclidean(rho[i], theta[i])
    print(a.size())

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
