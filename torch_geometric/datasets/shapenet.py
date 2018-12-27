import os
import os.path as osp
import glob

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.read import read_txt_array

from ..data.makedirs import makedirs


class ShapeNet(InMemoryDataset):
    r""" The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (string): Root directory where the dataset should be saved.
        category (string): The category of the CAD models (one of
            :obj:`"Airplane"`, :obj:`"Bag"`, :obj:`"Cap"`, :obj:`"Car"`,
            :obj:`"Chair"`, :obj:`"Earphone"`, :obj:`"Guitar"`,
            :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`,
            :obj:`"Rocket"`, :obj:`"Skateboard"`, :obj:`"Table"`).
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://shapenet.cs.stanford.edu/iccv17/partseg'

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

    def __init__(self,
                 root,
                 category,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert category in self.categories.keys()
        self.category = category
        super(ShapeNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train_data', 'train_label', 'val_data', 'val_label', 'test_data',
            'test_label'
        ]

    @property
    def processed_file_names(self):
        names = ['training.pt', 'test.pt']
        return [osp.join(self.category, name) for name in names]

    def download(self):
        for name in self.raw_file_names:
            url = '{}/{}.zip'.format(self.url, name)
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        idx = self.categories[self.category]
        paths = [osp.join(path, idx) for path in self.raw_paths]
        datasets = []
        for path in zip(paths[::2], paths[1::2]):
            pos_paths = sorted(glob.glob(osp.join(path[0], '*.pts')))
            y_paths = sorted(glob.glob(osp.join(path[1], '*.seg')))
            data_list = []
            for path in zip(pos_paths, y_paths):
                pos = read_txt_array(path[0])
                y = read_txt_array(path[1], dtype=torch.long)
                data = Data(y=y, pos=pos)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            datasets.append(data_list)

        makedirs(osp.join(self.processed_dir, self.category))
        data_list = datasets[0] + datasets[1]
        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(self.collate(datasets[2]), self.processed_paths[1])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__,
                                            len(self), self.category)
