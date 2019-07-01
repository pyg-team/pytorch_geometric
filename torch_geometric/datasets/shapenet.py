import os
import os.path as osp
import glob

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.read import read_txt_array


class ShapeNet(InMemoryDataset):
    r""" The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitely set to :obj:`None` to load all categories.
            (default: :obj:`None`)
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

    category_ids = {
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
                 categories=None,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super(ShapeNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        self.y_mask = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return [
            'train_data', 'train_label', 'val_data', 'val_label', 'test_data',
            'test_label'
        ]

    @property
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            '{}_{}.pt'.format(cats, s) for s in ['training', 'test', 'y_mask']
        ]

    def download(self):
        for name in self.raw_file_names:
            url = '{}/{}.zip'.format(self.url, name)
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process_raw_path(self, data_path, label_path):
        y_offset = 0
        data_list = []
        cat_ys = []
        for cat_idx, cat in enumerate(self.categories):
            idx = self.category_ids[cat]
            point_paths = sorted(glob.glob(osp.join(data_path, idx, '*.pts')))
            y_paths = sorted(glob.glob(osp.join(label_path, idx, '*.seg')))

            points = [read_txt_array(path) for path in point_paths]
            ys = [read_txt_array(path, dtype=torch.long) for path in y_paths]
            lens = [y.size(0) for y in ys]

            y = torch.cat(ys).unique(return_inverse=True)[1] + y_offset
            cat_ys.append(y.unique())
            y_offset = y.max().item() + 1
            ys = y.split(lens)

            for (pos, y) in zip(points, ys):
                data = Data(y=y, pos=pos, category=cat_idx)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        y_mask = torch.zeros((len(self.categories), y_offset),
                             dtype=torch.uint8)
        for i in range(len(cat_ys)):
            y_mask[i, cat_ys[i]] = 1

        return data_list, y_mask

    def process(self):
        train_data_list, y_mask = self.process_raw_path(*self.raw_paths[0:2])
        val_data_list, _ = self.process_raw_path(*self.raw_paths[2:4])
        test_data_list, _ = self.process_raw_path(*self.raw_paths[4:6])

        data = self.collate(train_data_list + val_data_list)
        torch.save(data, self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
        torch.save(y_mask, self.processed_paths[2])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)
