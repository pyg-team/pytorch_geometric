import os.path as osp

import torch
from torch_sparse import SparseTensor
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, sparse=False, transform=None,
                 pre_transform=None):
        self.name = name
        super(Planetoid, self).__init__(root, transform, pre_transform)
        self.__data__ = torch.load(self.processed_paths[0])
        if sparse:
            x = SparseTensor.from_dense(self.__data__.x, has_value=False)
            self.__data__.x = x.fill_cache_()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0 or idx == -1
        data = self.__data__
        if self.transform is not None:
            data = self.transform(data)
        return data

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
