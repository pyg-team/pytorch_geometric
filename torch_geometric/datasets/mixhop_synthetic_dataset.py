import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class MixHopSyntheticDataset(InMemoryDataset):
    r"""The MixHop synthetic dataset from the `"MixHop: Higher-Order
    Graph Convolutional Architectures via Sparsified Neighborhood Mixing"
    <https://arxiv.org/abs/1905.00067>`_ paper, containing 10
    graphs, each with varying degree of homophily (ranging from 0.0 to 0.9).
    All graphs have 5,000 nodes, where each node corresponds to 1 out of 10
    classes.
    The feature values of the nodes are sampled from a 2D Gaussian
    distribution, which are distinct for each class.

    Args:
        root (str): Root directory where the dataset should be saved.
        homophily (float): The degree of homophily (one of :obj:`0.0`,
            :obj:`0.1`, ..., :obj:`0.9`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = ('https://raw.githubusercontent.com/samihaija/mixhop/master/data'
           '/synthetic')

    def __init__(
        self,
        root: str,
        homophily: float,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.homophily = homophily
        assert homophily in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, f'{self.homophily:0.1f}'[::2], 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.homophily:0.1f}'[::2], 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        name = f'ind.n5000-h{self.homophily:0.1f}-c10'
        return [f'{name}.allx', f'{name}.ally', f'{name}.graph']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        x = torch.from_numpy(np.load(self.raw_paths[0]))
        y = torch.from_numpy(np.load(self.raw_paths[1])).argmax(dim=-1)

        edges = pickle.load(open(self.raw_paths[2], 'rb'), encoding='latin1')
        row, col = [], []
        for k, v in edges.items():
            row += [k] * len(v)
            col += v

        edge_index = torch.tensor([row, col], dtype=torch.long)

        N_s = x.size(0) // 3
        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[:N_s] = True
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[N_s:2 * N_s] = True
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[2 * N_s:] = True

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(homophily={self.homophily:.1f})'
