import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz


class CitationFull(InMemoryDataset):
    r"""The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    Datasets include `citeseer`, `cora`, `cora_ml`, `dblp`, `pubmed`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
            :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed']
        super(CitationFull, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}Full()'.format(self.name.capitalize())


class CoraFull(CitationFull):
    r"""Alias for :class:`torch_geometric.dataset.CitationFull` with
    :obj:`name="cora"`."""
    def __init__(self, root, transform=None, pre_transform=None):
        super(CoraFull, self).__init__(root, 'cora', transform, pre_transform)

    def download(self):
        super(CoraFull, self).download()

    def process(self):
        super(CoraFull, self).process()
