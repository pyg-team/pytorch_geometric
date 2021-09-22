from typing import Optional, Callable, List

import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets, as pre-processed by
    `Renchi Yang <https://renchi.ac.cn/datasets/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`). # TODO
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://entuedu-my.sharepoint.com/personal/yang0461_e_ntu_edu_sg/'
           'Documents/shared-data')

    # url = ('https://entuedu-my.sharepoint.com/personal/yang0461_e_ntu_edu_sg/'
    #        '_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fyang0461%5Fe%'
    #        '5Fntu%5Fedu%5Fsg%2FDocuments%2Fshared%2Ddata%{}%2Eattr%2Etar%2Egz')

    names = ['wiki', 'cora', 'citeseer']

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.names
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return 'dawd'
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = f'{self.url}/cora.attr.tar.gz'
        print(url)
        download_url(url, self.raw_dir)
        # for name in self.raw_file_names:
        #     download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        raise NotImplementedError
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
