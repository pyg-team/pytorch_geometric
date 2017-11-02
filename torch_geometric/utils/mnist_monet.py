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


def sparse_to_dict(a):
    index = a._indices()
    x, y = a._values().t()

    d = {}
    for i in range(index.size(1)):
        row = index[0, i]
        if row not in d:
            d[row] = []
        d[row].append((index[1, i], x[i], y[i]))
    return d


def dict_to_pos(a):
    pos = torch.FloatTensor(75, 2).fill_(0)

    parent = [0 for _ in a[0]]
    to_process = [c for c in a[0]]
    processed = [0]

    while len(to_process) > 0:
        r = parent.pop(0)
        data = to_process.pop(0)
        c, x, y = data

        r_pos = pos[r]
        pos[c, 0] = r_pos[0] + x
        pos[c, 1] = r_pos[1] + y

        processed.append(c)

        new_edges = list(filter(lambda x: x[0] not in processed, a[c]))
        parent.extend([c for _ in new_edges])
        to_process.extend([i for i in new_edges])

    return pos


path = os.path.expanduser('~/Downloads/MoNet_code/MNIST')
# data = read_mnist_monet(os.path.join(path, 'datasets'))
# torch.save(data, os.path.join(path, 'data.pt'))
# data = torch.load(os.path.join(path, 'data.pt'))
# input, target, rho, theta = data
# target = target.byte()

# position = torch.load(os.path.join(path, 'positions.pt'))

# indices = []
# for i in range(70000):
#     nz = rho[i].nonzero().t().byte()
#     indices.append(nz)

# positions = []
# for i in range(70000):
#     adj = polar2euclidean(rho[i], theta[i])
#     adj = sparse_to_dict(adj)
#     pos = dict_to_pos(adj)
#     minimum, _ = pos.min(dim=0)
#     pos = pos - minimum
#     positions.append(pos)

# positions = torch.stack(positions, dim=0)

# torch.save(positions, os.path.join(path, 'positions.pt'))

# data = torch.load(os.path.join(path, 'data.pt'))
# input, position, index, slice, target = data

# train_input = input[:60000].contiguous()
# train_position = position[:60000].contiguous()
# train_target = target[:60000].contiguous()
# test_input = input[60000:].contiguous()
# test_position = position[60000:].contiguous()
# test_target = target[60000:].contiguous()
# train_slice = slice[:60001].contiguous()
# test_slice = slice[60000:].contiguous()
# test_slice = test_slice - test_slice[0]
# train_index = index[:, :(train_slice[-1])].contiguous()
# test_index = index[:, train_slice[-1]:].contiguous()

# torch.save((train_input, train_index, train_slice, train_position,
#             train_target), os.path.join(path, 'training.pt'))
# torch.save((test_input, test_index, test_slice, test_position, test_target),
#            os.path.join(path, 'test.pt'))
# path = os.path.expanduser('~/MNISTSuperpixel75')
# train_data = torch.load(os.path.join(path, 'test.pt'))

# input, index, slice, position, target = train_data

# pos_y = position[:, :, 0]
# pos_x = position[:, :, 1]
# pos_y_max, _ = pos_y.max(dim=1)
# pos_x_max, _ = pos_x.max(dim=1)
# pos_y = pos_y_max.view(-1, 1) - pos_y
# pos_x = pos_x_max.view(-1, 1) - pos_x
# print(pos_x.size())

# position = torch.stack([pos_x, pos_y], dim=2)
# torch.save((input, index, slice, position, target),
#            os.path.join(path, 'test.pt'))
