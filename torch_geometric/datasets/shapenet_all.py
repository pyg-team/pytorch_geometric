import os
import os.path as osp
import glob

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)
from torch_geometric.read import read_txt_array
from torch_geometric.data.makedirs import makedirs

class ShapeNet(InMemoryDataset):
    r""" The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): Data split, options: train, val, test.
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
                 split,
                 classification=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform,
                                       pre_filter)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]    
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train_data', 'train_label', 
                'val_data', 'val_label', 
                'test_data', 'test_label']

    @property
    def processed_file_names(self):
        return ['all_train.pt', 'all_val.pt', 'all_test.pt']

    @property
    def num_categories(self):
        r"""The number of categories in the dataset."""
        return len(self.categories)

    def download(self):
        for name in self.raw_file_names:
            url = '{}/{}.zip'.format(self.url, name)
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
            
    def reindex_labels(self, y, cat_sid):
        '''Re-index original labels to counts all categories'''
        ymin = y.min()
        y = y - ymin # Move to 0-based indexing
        ymax = y.max()
        y += cat_sid
        assert y.max() == (cat_sid + ymax )
        assert y.min() == cat_sid
        return y
    
    def process(self):
        data_splits = [[], [], []]  # Corresponds to train/val/test
        cat_names = list(self.categories.keys())
        category_infos = {}
        cat_sid = 0
        for cid, cat in enumerate(cat_names):
            infos = {'category_id':cid}
            paths = [osp.join(path, self.categories[cat])for path in self.raw_paths]
            print('Processing category {}'.format(cat))

            # Loop over train/val/test split
            ymax = 0  # Largest label value of this category
            sample_nums = []
            for split, path in enumerate(zip(paths[::2], paths[1::2])):
                pos_paths = sorted(glob.glob(osp.join(path[0], '*.pts')))
                y_paths = sorted(glob.glob(osp.join(path[1], '*.seg')))

                # Gather all category split samples
                data_list = []
                for path in zip(pos_paths, y_paths):
                    pos = read_txt_array(path[0])
                    y = read_txt_array(path[1], dtype=torch.long)
                    y = self.reindex_labels(y, cat_sid)  
                    ymax = max(ymax, y.max())
                    data = Data(y=y, pos=pos, cid=cid)

                    # Data preprocessing
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                sample_nums.append(len(data_list))
                data_splits[split] += data_list # Merge splits data across categories
            cat_eid = ymax.item()
            infos['part_id_min'] = cat_sid
            infos['part_id_max'] = cat_eid
            infos['part_num'] = cat_eid - cat_sid + 1
            infos['sample_num'] = sample_nums
            category_infos[cat] = infos
            print('Infos:{}'.format(infos))
            cat_sid = cat_eid + 1 # Update category start index

        train_data, train_slices = self.collate(data_splits[0])
        val_data, val_slices = self.collate(data_splits[1])
        test_data, test_slices = self.collate(data_splits[2])

        # Save processed data
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((val_data, val_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])
        torch.save(category_infos, osp.join(self.processed_dir, 'all_infos.pt'))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))