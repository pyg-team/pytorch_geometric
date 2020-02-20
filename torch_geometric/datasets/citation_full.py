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
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        if name not in ['citeseer', 'cora', 'cora_ml', 'dblp', 'pubmed']:
            print("Cannot find the dataset {}".format(name))
            raise IOError
        self.name = name
        _url = "'https://github.com/abojchevski/graph2gauss/raw/master/data/"
        self.url = _url + '{}.npz'.format(self.name)
        super(CitationFull, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
