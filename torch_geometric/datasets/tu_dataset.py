import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_tu_data


class TUDataset(InMemoryDataset):
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.name = name
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        try:
            self.data, self.slices, self.props = torch.load(self.processed_paths[0])
        except ValueError as e:
            raise type(e)(str(e) + '. This can be probably solved by removing old dataset file %s and running the code again.' % self.processed_paths[0])

        if use_node_attr:
            if self.props['node_attr_dim'] == 0:
                raise ValueError('Continuous node attributes are not found in this dataset. Try running with use_node_attr=False.')
        else:
            self.data.x = self.data.x[:, self.props['node_attr_dim']:]  # omit continuous node attributes

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, self.props = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices, self.props), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
