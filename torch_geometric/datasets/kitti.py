import os
import os.path as osp
import shutil

import torch
import numpy as np

from .dataset import Dataset, Data, _exists
from .utils.download import download_url
from .utils.extract import extract_zip, extract_tar
from .utils.dir import make_dirs
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
        self.train = train

    @property
    def raw_files(self):
        return ['training/velodyne', 'training/label_2', 'ImageSets']

    @property
    def processed_files(self):
        return [self.processed_folder+'/train/'+'example'+str(i)+'.pt'
                for i in range(100)] + \
               [self.processed_folder+'/test/'+'example'+str(i)+'.pt'
                for i in range(100)]

    def download(self):
        file_path = download_url(self.data_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)
        shutil.rmtree(osp.join(self.raw_folder, 'testing'))

        file_path = download_url(self.target_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

        file_path = download_url(self.split_url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)

    def _process(self):
        if _exists(self._processed_files):
            return

        make_dirs(self.processed_folder)
        with open(osp.join(self.raw_folder, 'ImageSets/train.txt'), 'r') as f:
            train_files = f.read().split()
        with open(osp.join(self.raw_folder, 'ImageSets/val.txt'), 'r') as f:
            test_files = f.read().split()

        # num_examples = len(train_files) + len(test_files)
        num_examples = 7481
        progress = Progress('Processing', end=num_examples, type='')
        make_dirs(self.processed_folder + '/train/')
        make_dirs(self.processed_folder + '/test/')
        for i in range(3712):
            data = self.process_example(self.raw_folder, train_files[i])
            progress.inc()
            make_dirs(self.processed_folder + '/train/')
            torch.save(
                data,
                self.processed_folder + '/train/' + 'example' + str(i) + '.pt')
        for i in range(3769):
            data = self.process_example(self.raw_folder, test_files[i])
            progress.inc()
            torch.save(
                data,
                self.processed_folder + '/test/' + 'example' + str(i) + '.pt')

    def process_example(self, dir, name):
        filename = osp.join(dir, 'training', 'velodyne', '{}.bin'.format(name))
        scan = torch.from_numpy(np.fromfile(filename, dtype=np.float32))
        scan = scan.view(-1, 4)  # [x, y, z, reflectance]
        mask = (scan[:, 2] <= 1) * (scan[:, 2] >= -3) * (scan[:, 1] >= -40) * \
               (scan[:, 1] <= 40) * (scan[:, 0] >= 0) * (scan[:, 0] <= 70.4)

        scan = scan[mask.nonzero(), :].squeeze()
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

    def __getitem__(self, i):
        traintest = 'train' if self.train else 'test'
        data = torch.load(self.processed_folder + '/' + traintest + '/' +
                          'example' + str(i) + '.pt')
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        if self.train:
            return 3712
        else:
            return 3769
