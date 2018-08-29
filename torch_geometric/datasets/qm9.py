import os

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)


class QM9(InMemoryDataset):
    url = 'http://www.roemisch-drei.de/qm9.tar.gz'

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        extract_tar(file_path, self.raw_dir, mode='r')
        os.unlink(file_path)

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                y=d['y'],
                pos=d['pos']) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
