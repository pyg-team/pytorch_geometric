import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class EllipticBitcoinDataset(InMemoryDataset):
    r"""The Elliptic Bitcoin dataset of Bitcoin transactions from the
    `"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
    Networks for Financial Forensics" <https://arxiv.org/abs/1908.02591>`_
    paper.

    :class:`EllipticBitcoinDataset` maps Bitcoin transactions to real entities
    belonging to licit categories (exchanges, wallet providers, miners,
    licit services, etc.) versus illicit ones (scams, malware, terrorist
    organizations, ransomware, Ponzi schemes, etc.)

    There exists 203,769 node transactions and 234,355 directed edge payments
    flows, with two percent of nodes (4,545) labelled as illicit, and
    twenty-one percent of nodes (42,019) labelled as licit.
    The remaining transactions are unknown.

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

    Stats:
        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 203,769
              - 234,355
              - 165
              - 2
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url('https://tinyurl.com/9b7f8efe', self.raw_dir)
        os.rename(osp.join(self.raw_dir, '9b7f8efe'), self.raw_paths[0])
        download_url('https://tinyurl.com/mr3v9d3f', self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'mr3v9d3f'), self.raw_paths[1])
        download_url('https://tinyurl.com/2p8up25z', self.raw_dir)
        os.rename(osp.join(self.raw_dir, '2p8up25z'), self.raw_paths[2])

    def process(self):
        import pandas as pd

        df_features = pd.read_csv(self.raw_paths[0], header=None)
        df_edges = pd.read_csv(self.raw_paths[1])
        df_classes = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        df_features = df_features.rename(columns=columns)
        x = torch.from_numpy(df_features.loc[:, 2:].values).to(torch.float)

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        mapping = {'unknown': 2, '1': 1, '2': 0}
        df_classes['class'] = df_classes['class'].map(mapping)
        y = torch.from_numpy(df_classes['class'].values)

        mapping = {idx: i for i, idx in enumerate(df_features['txId'].values)}
        df_edges['txId1'] = df_edges['txId1'].map(mapping)
        df_edges['txId2'] = df_edges['txId2'].map(mapping)
        edge_index = torch.from_numpy(df_edges.values).t().contiguous()

        # Timestamp based split:
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = torch.from_numpy(df_features['time_step'].values)
        train_mask = (time_step) < 35 & (y != 2)
        test_mask = (time_step) >= 35 & (y != 2)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 2
