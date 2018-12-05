import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.read import read_ply


class CoMA(InMemoryDataset):
    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(CoMA, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'COMA_data.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download COMA_data.zip from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))
        if len(folders) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))

        train_data_list, test_data_list = [], []
        for folder in folders:
            for i, category in enumerate(self.categories):
                files = sorted(glob(osp.join(folder, category, '*.ply')))
                for j, f in enumerate(files):
                    data = read_ply(f)
                    data.y = torch.tensor([i], dtype=torch.long)
                    if self.pre_filter is not None and\
                       not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if (j % 100) < 90:
                        train_data_list.append(data)
                    else:
                        test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
