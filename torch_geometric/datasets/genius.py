import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class GeniusDataset(InMemoryDataset):
    r"""The 'genius' datasets from  `"Expertise and dynamics within
    crowdsourced musical knowledge curation"
    <https://arxiv.org/abs/2006.08108>`_ paper.

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

    url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/' \
          'data/genius.mat'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert self.name in self.datasets.keys()
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
        names = [self.datasets[self.name].split('/')[-1]]
        if self.name in self.splits:
            names += [self.splits[self.name].split('/')[-1]]
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.datasets[self.name], self.raw_dir)
        if self.name in self.splits:
            download_url(self.splits[self.name], self.raw_dir)

    def process(self):
        from scipy.io import loadmat

        fulldata = loadmat(self.raw_paths[0])
        edge_index = torch.tensor(fulldata['edge_index'].astype('int64'))
        x = torch.tensor(fulldata['node_feat'], dtype=torch.float)
        y = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()

        data = Data(x=x, edge_index=edge_index, y=y)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}({len(self)})'
