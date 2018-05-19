import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_tu_dataset


class TUDataset(InMemoryDataset):
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets/'

    def __init__(self, root, name, transform=None):
        self.name = name
        super(TUDataset, self).__init__(root, transform)
        self.dataset, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator', 'graph_labels', 'node_labels']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = '{}/{}.zip'.format(self.url, self.name)
        path = download_url(url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        data_list = read_tu_dataset(self.raw_dir)
        dataset, slices = self.collate(data_list)
        torch.save((dataset, slices), self.processed_paths[0])
