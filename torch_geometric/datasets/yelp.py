import json
import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class Yelp(InMemoryDataset):
    r"""The Yelp dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing customer reviewers and their friendship.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #tasks
        * - 716,847
          - 13,954,819
          - 300
          - 100
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    adj_full_id = '1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA'
    feats_id = '1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM'
    class_map_id = '1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU'
    role_id = '1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url.format(self.adj_full_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'adj_full.npz'))

        path = download_url(self.url.format(self.feats_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'feats.npy'))

        path = download_url(self.url.format(self.class_map_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'class_map.json'))

        path = download_url(self.url.format(self.role_id), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'role.json'))

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
