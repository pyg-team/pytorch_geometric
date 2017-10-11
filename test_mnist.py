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


def load_matlab_array(path_file, name_field, dtype=np.float32):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    return np.asarray(ds).astype(dtype).T.squeeze()


# a = load_matlab_array(path_train_vals, 'vals')
# b = load_matlab_array(path_test_vals, 'vals')
# c = load_matlab_array(path_train_coords, 'patch_coords')
# d = load_matlab_array(path_test_coords, 'patch_coords')
# 75 * 75 * 2 values (warum 2?)
e = load_matlab_array(path_train_centroids, 'idx_centroids')
f = load_matlab_array(path_test_centroids, 'idx_centroids')
# print(e.shape)
# print(e)

train_labels = load_matlab_array(path_train_labels, 'labels', np.uint8)
test_labels = load_matlab_array(path_test_labels, 'labels', np.uint8)
print(test_labels.dtype)
print(test_labels.shape)
print(test_labels)

# print(path_train_vals)
# path_test_vals = os.path.join(
#     path_main, 'datasets/mnist_superpixels_data_%d/test_vals.mat' % n_supPix)
# # path to the patches
# path_coords_train = os.path.join(
#     path_main,
#     'datasets/mnist_superpixels_data_%d/train_patch_coords.mat' % n_supPix)
# path_coords_test = os.path.join(
#     path_main,
#     'datasets/mnist_superpixels_data_%d/test_patch_coords.mat' % n_supPix)
# # path to the labels
# path_train_labels = os.path.join(
#     path_main, 'datasets/MNIST_preproc_train_labels/MNIST_labels.mat')
# path_test_labels = os.path.join(
#     path_main, 'datasets/MNIST_preproc_test_labels/MNIST_labels.mat')

# # path to the idx centroids
# path_train_centroids = os.path.join(
#     path_main,
#     'datasets/mnist_superpixels_data_%d/train_centroids.mat' % n_supPix)
# path_test_centroids = os.path.join(
#     path_main,
#     'datasets/mnist_superpixels_data_%d/test_centroids.mat' % n_supPix)
