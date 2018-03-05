import os
import os.path as osp
import json

import torch

from .dataset import Dataset, Data
from .utils.download import download_url
from .utils.extract import extract_zip
from .utils.progress import Progress
from .utils.nn_graph import nn_graph


class ShapeNet(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/' \
          'shapenetcore_partanno_segmentation_benchmark_v0.zip'

    categories = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    def __init__(self, root, category, train=True, transform=None):
        assert category in self.categories, 'No valid category'
        self.id = self.categories[category]

        super(ShapeNet, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        return 'shape_data'

    @property
    def processed_files(self):
        return ['{}_training.pt'.format(self.id), '{}_test.pt'.format(self.id)]

    def _rename(self, a, b):
        os.rename(osp.join(self.raw_folder, a), osp.join(self.raw_folder, b))

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)
        dir = 'shapenetcore_partanno_segmentation_benchmark_v0'
        self._rename(dir, 'shape_data')

    def process(self):
        # Read and conver test_split to identifier list.
        split_dir = osp.join(self.raw_folder, 'shape_data', 'train_test_split')
        test_split_file = osp.join(split_dir, 'shuffled_test_file_list.json')
        test_split = json.load(open(test_split_file, 'r'))
        p = 'shape_data/{}/'.format(self.id)
        test_split = [x.split(p)[1] for x in test_split if x.startswith(p)]
        test_split = sorted(test_split)

        # Save input filenames.
        category_dir = osp.join(self.raw_folder, 'shape_data', self.id)
        pos_dir = osp.join(category_dir, 'points')
        target_dir = osp.join(category_dir, 'points_label')
        ids = sorted([f.split('.')[0] for f in os.listdir(pos_dir)])

        # Convert examples and append either to training or test set.
        progress = Progress('Processing', end=len(ids), type='')

        train_set, test_set = [], []
        cur = 0
        for id in ids:
            data = self.process_example(pos_dir, target_dir, id)
            if cur < len(test_split) and id == test_split[cur]:
                cur += 1
                test_set.append(data)
            else:
                train_set.append(data)
            progress.inc()
        progress.success()

        return train_set, test_set

    def process_example(self, pos_dir, target_dir, id):
        # Read three-dimensional positions.
        with open(osp.join(pos_dir, '{}.pts'.format(id)), 'r') as f:
            pos = [float(x) for x in f.read().split()]
            pos = torch.FloatTensor(pos).view(-1, 3)

        # Read targets.
        with open(osp.join(target_dir, '{}.seg'.format(id)), 'r') as f:
            target = [int(x) for x in f.read().split()]
            target = torch.LongTensor(target)

        # No input features are given.
        input = pos.new(pos.size(0)).fill_(1)

        # 6-nearest neighbor for edge indices.
        index = nn_graph(pos, k=6)

        return Data(input, pos, index, None, target)
