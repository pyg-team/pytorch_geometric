import os
import scipy.io as spio

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_zip
from ..graph.point_cloud import nn_graph


class ShapeNet(Dataset):
    url = ('https://shapenet.cs.stanford.edu/ericyi/scnn_data.zip')

    def __init__(self,
                 root,
                 train=True,
                 category='Airplaine',
                 transform=None,
                 target_transform=None):

        root = os.path.expanduser(root)
        self.processed_folder = os.path.join(root, 'processed')
        self.raw_folder = os.path.join(root, 'raw')
        self.category = category

        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

        filename = 'training.pt' if train else 'test.pt'
        data_file = os.path.join(self.processed_folder, category, filename)
        data = torch.load(data_file)
        self.input, self.index, self.point, self.target, self.slice = data
        self.target = self.target.long() - 1

    def __getitem__(self, i):
        start, end = self.slice[i:i + 2]

        input = self.input[start:end]
        index = self.index[:, 6 * start:6 * end]
        n = input.size(0)
        weight = torch.FloatTensor(index.size(1)).fill_(1)
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))
        point = self.point[start:end]
        target = self.target[start:end]
        data = (input, adj, point)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.slice.size(0) - 1

    def _check_exists(self):
        return os.path.exists(self.raw_folder)

    def _check_processed(self):
        return os.path.exists(self.processed_folder)

    @property
    def categories(self):
        with open(os.path.join(self.raw_folder, 'Data/CategoryList.txt')) as f:
            return f.read().split('\n')[:-1]

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(self.processed_folder)

        for category in self.categories:
            self._process_category(category)

        print('Done!')

    def _process_category(self, category):
        raw_folder = os.path.join(self.raw_folder, 'Data/Categories', category)
        processed_folder = os.path.join(self.processed_folder, category)
        make_dirs(processed_folder)

        data = self._read_file(os.path.join(raw_folder, 'train_1.mat'))
        t_input, t_index, t_point, t_target, t_slice = data

        data = self._read_file(os.path.join(raw_folder, 'val_1.mat'))
        v_input, v_index, v_point, v_target, v_slice = data
        v_slice += t_slice[-1]

        input = torch.cat([t_input, v_input], dim=0)
        index = torch.cat([t_index, v_index], dim=1)
        point = torch.cat([t_point, v_point], dim=0)
        target = torch.cat([t_target, v_target], dim=0)
        slice = torch.cat([t_slice, v_slice[1:]], dim=0)
        data = (input, index, point, target, slice)
        torch.save(data, os.path.join(processed_folder, 'training.pt'))

        data = self._read_file(os.path.join(raw_folder, 'test_1.mat'))
        torch.save(data, os.path.join(processed_folder, 'test.pt'))

    def _read_file(self, path):
        data = spio.loadmat(path)
        input, point, target = data['v'], data['pts'], data['label']

        slice = [0]
        for t in target:
            slice.append(slice[-1] + len(t[0]))
        slice = torch.LongTensor(slice)

        input = torch.cat([torch.FloatTensor(i[0]) for i in input], dim=0)
        points = [torch.FloatTensor(p[0]) for p in point]
        indices = [nn_graph(p, k=6) for p in points]
        point = torch.cat(points, dim=0)
        index = torch.cat(indices, dim=1)
        target = torch.cat([torch.FloatTensor(t[0]) for t in target], dim=0)
        target = target.byte().view(-1)

        return input, index, point, target, slice
