import os
import os.path as osp
import shutil
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_gz, extract_tar, extract_zip
from torch_geometric.data.makedirs import makedirs


def read_stanford_data(folder, name, file):
    if 'ego-' in name:
        folder = osp.join(folder, file[:-7])
        files = [file for file in os.listdir(folder) if osp.isfile(osp.join(folder, file))]
        files.sort()

        #edge_index, x = torch.Tensor().long(), torch.Tensor().long()
        edge_index, x = [], []
        for file in files:
            if file.endswith('circles'):
                pass

            elif file.endswith('edges'):
                _edge_index = torch.from_numpy(np.loadtxt(osp.join(folder, file)))\
                              .transpose(0,1).long()
                # edge_index = torch.cat((edge_index, _edge_index), dim=1)
                edge_index.append(_edge_index)

            elif file.endswith('egofeat'):
                _x_ego = torch.from_numpy(np.loadtxt(osp.join(folder, file))).long()
                _x_ego = torch.cat((torch.zeros(1).long(), _x_ego)).unsqueeze(0)  # id: 0 for egoNode

            elif file.endswith('feat'):
                _x = torch.from_numpy(np.loadtxt(osp.join(folder, file))).long()
                _x = torch.cat((_x, _x_ego)) 
                # x = torch.cat((x, _x))
                x.append(_x)

            elif file.endswith('featnames'):
                print(file)


        print([(i.shape,j.shape) for i,j in zip(x, edge_index)])
        raise NotImplementedError


class StanfordDataset(InMemoryDataset):

    def __init__(self, root,  name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False):

        self.url = 'https://snap.stanford.edu/data'
        self.available_datasets_file = {'ego-Facebook': 'facebook.tar.gz',
                                        'ego-Gplus': 'gplus.tar.gz',
                                        'ego-Twitter': 'twitter.tar.gz'}
        available_datasets = list(self.available_datasets_file.keys())

        if name not in available_datasets:
            raise NameError('Name \'{}\' is not an available dataset. Try one of these {}.'.format(name, available_datasets))

        self.name = name
        super(StanfordDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]


    def _download(self):
        if not osp.isdir(self.raw_dir):
            makedirs(self.raw_dir)
            self.download()
        elif len(os.listdir(self.raw_dir)) == 0:
            self.download()
        else:
            return


    def download(self):
        folder = osp.join(self.root, self.name)
        if 'ego-' in self.name:
            path = download_url('{}/{}'.format(self.url, self.available_datasets_file[self.name]), folder)
            extract_tar(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)


    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        self.data, self.slices = read_stanford_data(self.raw_dir, self.name, self.available_datasets_file[self.name])

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


if __name__ == '__main__':
    dataset_name = 'ego-Facebook'
    path = path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', dataset_name)
    dataset = StanfordDataset(path, dataset_name)


