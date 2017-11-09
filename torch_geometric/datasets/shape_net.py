import os
import json

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_zip


class ShapeNet(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'
    raw_folder = 'shapenetcore_partanno_segmentation_benchmark_v0'

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
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

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
        # if self._check_processed():
        #     return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))

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

        train_points = []
        train_targets = []
        test_points = []
        test_targets = []

        for filename in os.listdir(point_dir):
            idx = filename.split('.')[0]

            point_file = os.path.join(point_dir, '{}.pts'.format(idx))
            with open(point_file, 'r') as f:
                points = [float(x) for x in f.read().split()]
                point = torch.FloatTensor(points).view(-1, 3)

            target_file = os.path.join(target_dir, '{}.seg'.format(idx))
            with open(target_file, 'r') as f:
                targets = [int(x) for x in f.read().split()]
                target = torch.ByteTensor(targets)

            print(target.size(), point.size())
            if idx not in test_indices:
                train_points.append(point)
                train_targets.append(target)
            else:
                test_points.append(point)
                test_targets.append(target)

            # TODO: Generate edges
            # TODO: Generate weights

            # TODO: Save as *.pt
