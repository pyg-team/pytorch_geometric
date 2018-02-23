import os
import os.path as osp
import shutil

import torch
import numpy as np

from .dataset import Dataset, Data
from .utils.download import download_url
from .utils.extract import extract_zip, extract_tar
from .utils.progress import Progress
from .utils.nn_graph import nn_graph


class KITTI(Dataset):
    """`KITTI Vision Benchmark <http://www.cvlibs.net/datasets/kitti>`_
    Velodyne Point Cloud Dataset from the
    `"Are we ready for Autonous Driving? The KITTI Vision Benchmark Suite"
    <http://www.cvlibs.net/publications/Geiger2012CVPR.pdf>`_ paper. KITTI
    consists of 7,481 training and 7,518 test point clouds, covering three
    categories: Car, Pedestrian, and Cyclist.

    Args:
        root (string): Root directory of dataset. Downloads data automatically
            if dataset doesn't exist.
        train (bool, optional): If :obj:`True`, returns training data,
            otherwise test data. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    data_url = 'http://kitti.is.tue.mpg.de/kitti/data_object_velodyne.zip'
    target_url = 'http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip'
    split_url = 'http://www.cs.toronto.edu/objprop3d/data/ImageSets.tar.gz'

    types = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    def __init__(self, root, train=True, transform=None):
        super(KITTI, self).__init__(root, transform)

    @property
    def raw_files(self):
        return ['training/velodyne', 'training/label_2', 'ImageSets']

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.data_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)
        shutil.rmtree(osp.join(self.raw_folder, 'testing'))
        shutil.rmtree(osp.join(self.raw_folder, 'training', 'det_2'))

        file_path = download_url(self.target_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

        file_path = download_url(self.split_url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        with open(osp.join(self.raw_folder, 'ImageSets/train.txt'), 'r') as f:
            train_files = f.read().split()
        with open(osp.join(self.raw_folder, 'ImageSets/val.txt'), 'r') as f:
            test_files = f.read().split()

        train, test = [], []
        # num_examples = len(train_files) + len(test_files)
        num_examples = 200
        progress = Progress('Processing', end=num_examples, type='')
        for i in range(100):
            train.append(self.process_example(self.raw_folder, train_files[i]))
            progress.inc()
        for i in range(100):
            test.append(self.process_example(self.raw_folder, test_files[i]))
            progress.inc()

        # return train, test

    def process_example(self, dir, name):
        filename = osp.join(dir, 'training', 'velodyne', '{}.bin'.format(name))
        scan = torch.from_numpy(np.fromfile(filename, dtype=np.float32))
        scan = scan.view(-1, 4)  # [x, y, z, reflectance]
        pos, input = scan[:, :3], scan[:, 3]
        index = None
        target = None
        index = nn_graph(pos)

        filename = osp.join(dir, 'training', 'label_2', '{}.txt'.format(name))
        with open(filename, 'r') as f:
            target = [t.split() for t in f.read().split('\n')[:-1]]
            target = [x for x in target if x[0] in self.types.keys()]
            for i in range(len(target)):
                target[i][0] = self.types[target[i][0]]
            target = [[float(x) for x in t] for t in target]
            target = torch.FloatTensor(target)
        # type, alpha, dimensions (h, w, d), location (x, y, z), rotation_y
        target = target[:, (0, 3, 8, 9, 10, 11, 12, 13, 14)]

        return Data(input, pos, index, None, target)
