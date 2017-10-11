from __future__ import division

import os
import h5py
import numpy as np

path_train_vals = os.path.join('./mnist/train_vals.mat')
path_test_vals = os.path.join('./mnist/test_vals.mat')
path_train_coords = os.path.join('./mnist/train_patch_coords.mat')
path_test_coords = os.path.join('./mnist/test_patch_coords.mat')
path_train_centroids = os.path.join('./mnist/train_centroids.mat')
path_test_centroids = os.path.join('./mnist/test_centroids.mat')
path_train_labels = os.path.join('./mnist/train_labels.mat')
path_test_labels = os.path.join('./mnist/test_labels.mat')


def load_matlab_file(path_file, name_field, dtype=np.float32):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    out = np.asarray(ds).astype(dtype).T.squeeze()
    db.close()
    return out


train_vals = load_matlab_file(path_train_vals, 'vals')
test_vals = load_matlab_file(path_test_vals, 'vals')
# train_coords = load_matlab_file(path_train_coords, 'patch_coords')
test_coords = load_matlab_file(path_test_coords, 'patch_coords')
train_centroids = load_matlab_file(path_train_centroids, 'idx_centroids',
                                   np.int64)
test_centroids = load_matlab_file(path_test_centroids, 'idx_centroids',
                                  np.int64)
train_labels = load_matlab_file(path_train_labels, 'labels', np.uint8)
test_labels = load_matlab_file(path_test_labels, 'labels', np.uint8)

# 75 * 75 * 2 values (warum 2?)
# 1-dim centroids? wtf?
# dist is 0 at root, rad is NaN


def stack_matrices(x):  # Stack 2D matrices into a 3D tensor
    n = int(np.sqrt(x.shape[1] // 2))
    num_examples = x.shape[0]

    rho = x[:, :n * n]
    theta = x[:, n * n:]
    rho = np.reshape(rho, (num_examples, n, n, 1))
    theta = np.reshape(theta, (num_examples, n, n, 1))
    return np.concatenate([rho, theta], axis=3)


# Load the coords
print(test_coords.shape)
test_coords = stack_matrices(test_coords)
print(test_coords.shape)
print(test_coords[0, 0, 0])
