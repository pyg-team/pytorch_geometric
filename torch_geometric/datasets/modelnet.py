import os
import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_off


class ModelNet(InMemoryDataset):
    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40':
        'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self,
                 root,
                 name='10',
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert name in ['10', '40']
        self.name = name
        super(ModelNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'ModelNet{}'.format(self.name))
        os.rename(folder, self.raw_dir)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split('/')[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
