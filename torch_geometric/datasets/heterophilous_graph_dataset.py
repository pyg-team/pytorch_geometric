import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected


class HeterophilousGraphDataset(InMemoryDataset):
    r"""The heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Roman-empire"`,
            :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"`,
            :obj:`"Questions"`).
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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Roman-empire
          - 22,662
          - 32,927
          - 300
          - 18
        * - Amazon-ratings
          - 24,492
          - 93,050
          - 300
          - 5
        * - Minesweeper
          - 10,000
          - 39,402
          - 7
          - 2
        * - Tolokers
          - 11,758
          - 519,000
          - 10
          - 2
        * - Questions
          - 48,921
          - 153,540
          - 301
          - 2
    """
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
        ]

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
