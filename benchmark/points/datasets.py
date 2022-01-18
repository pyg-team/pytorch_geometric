import os.path as osp

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T


def get_dataset(num_points):
    name = 'ModelNet10'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(path, name='10', train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name='10', train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset
