import os
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_zip
from ..graph.point_cloud import nn_graph
from ..transform.graclus import graclus, perm_input


class ShapeNet(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'
    raw_folder = 'shapenetcore_partanno_segmentation_benchmark_v0'
    training_filename = 'training.pt'
    test_filename = 'test.pt'

    categories = {
        'airplaine': '02691156',
        'bag': '02773838',
        'cap': '02954340',
        'car': '02958343',
        'chair': '03001627',
        'earphone': '03261776',
        'guitar': '03467517',
        'knife': '03624134',
        'lamp': '03636649',
        'laptop': '03642806',
        'bike': '03790512',
        'mug': '03797390',
        'pistol': '03948459',
        'rocket': '04099429',
        'skateboard': '04225987',
        'table': '04379243',
    }

    def __init__(self,
                 root,
                 train=True,
                 category='airplaine',
                 transform=None,
                 target_transform=None):

        super(ShapeNet, self).__init__()

        if category not in self.categories.keys():
            raise ValueError

        self.root = os.path.expanduser(root)
        self.processed_folder = os.path.join(self.root, 'processed')
        self.raw_folder = os.path.join(self.root, self.raw_folder)

        self.category = category
        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

        filename = self.training_filename if train else self.test_filename
        data_file = os.path.join(self.processed_folder, category, filename)
        self.index, self.point, target, self.slice = torch.load(data_file)
        self.target = target.long() - 1

    def __getitem__(self, i):
        start, end = self.slice[i:i + 2]
        index = self.index[:, 6 * start:6 * end]
        weight = torch.FloatTensor(index.size(1)).fill_(1)
        point = self.point[start:end]
        target = self.target[start:end]
        n = point.size(0)
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
        data = (input, adj, point)

        # adjs, points, perm = graclus(adj, point, level=2)
        # adj = adjs[2]
        # point = points[2]
        # n = point.size(0)
        # input = point.new(n, 1).fill_(1)

        # target = perm_input(target, perm).view(1, 1, -1)
        # target = F.max_pool1d(target.float(), 4).view(-1).data.long()

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.slice.size(0) - 1

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.processed_folder)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.root)
        extract_zip(file_path, self.root)
        os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(self.processed_folder)

        for category in self.categories:
            self._process(category)

        print('Done!')

    def _process_test_split(self, category, idx):
        filename = os.path.join(self.raw_folder, 'train_test_split',
                                'shuffled_test_file_list.json')
        with open(filename, 'r') as f:
            data = json.load(f)
            data = list(filter(lambda x: idx in x, data))
            return [x.split('/')[-1] for x in data]

    def _process(self, category):
        idx = self.categories[category]
        test_indices = self._process_test_split(category, idx)

        point_dir = os.path.join(self.raw_folder, idx, 'points')
        target_dir = os.path.join(self.raw_folder, idx, 'points_label')

        train_slices = [0]
        train_points = []
        train_targets = []
        train_edges = []
        test_slices = [0]
        test_points = []
        test_targets = []
        test_edges = []

        for i, filename in enumerate(os.listdir(point_dir)):
            idx = filename.split('.')[0]

            point_file = os.path.join(point_dir, '{}.pts'.format(idx))
            with open(point_file, 'r') as f:
                points = [float(x) for x in f.read().split()]
                point = torch.FloatTensor(points).view(-1, 3)

            index = nn_graph(point, k=6)

            target_file = os.path.join(target_dir, '{}.seg'.format(idx))
            with open(target_file, 'r') as f:
                targets = [int(x) for x in f.read().split()]
                target = torch.ByteTensor(targets)

            if idx not in test_indices:
                train_slices.append(train_slices[-1] + len(point))
                train_points.append(point)
                train_edges.append(index)
                train_targets.append(target)
            else:
                test_slices.append(test_slices[-1] + len(point))
                test_points.append(point)
                test_edges.append(index)
                test_targets.append(target)

        processed_folder = os.path.join(self.processed_folder, category)
        make_dirs(processed_folder)

        train_file = os.path.join(processed_folder, self.training_filename)
        slice = torch.LongTensor(train_slices)
        point = torch.cat(train_points, dim=0)
        index = torch.cat(train_edges, dim=1)
        target = torch.cat(train_targets, dim=0)
        torch.save((index, point, target, slice), train_file)

        test_file = os.path.join(processed_folder, self.test_filename)
        slice = torch.LongTensor(test_slices)
        point = torch.cat(test_points, dim=0)
        index = torch.cat(test_edges, dim=1)
        target = torch.cat(test_targets, dim=0)
        torch.save((index, point, target, slice), test_file)
