import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class EllipticBitcoinTemporalDataset(InMemoryDataset):
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
        t (int): Timestep for which nodes should be selected. Should be in
            the range 1-49 (both inclusive).
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
          - #classes
        * - 203,769
          - 234,355
          - 165
          - 2
    """
    url = 'https://data.pyg.org/datasets/elliptic'

    def __init__(self, t: int, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.t = t
        super().__init__(root, transform, pre_transform)
        print(f"loading data from {self.processed_paths[0]}")
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
        return f'data_v1_t={self.t}.pt'

    def download(self):
        for file_name in self.raw_file_names:
            path = download_url(f'{self.url}/{file_name}.zip', self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self):
        import pandas as pd

        t = self.t
        df_features = pd.read_csv(self.raw_paths[0], header=None)
        df_edges = pd.read_csv(self.raw_paths[1])
        df_classes = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        df_features = df_features.rename(columns=columns)

        # induce graph to nodes with timestep = t
        txId2timestep = dict()
        txId2timestepList = df_features[['txId',
                                         'time_step']].to_dict('records')
        for el in txId2timestepList:
            txId2timestep[el['txId']] = el['time_step']
        df_filtered_features = df_features[df_features['time_step'] == t]

        df_classes['time_step'] = df_classes.apply(
            lambda row: txId2timestep[row.txId], axis=1)
        df_filtered_classes = df_classes[df_classes['time_step'] == t].drop(
            'time_step', axis=1)

        df_edges['time_step'] = df_edges.apply(
            lambda row: max(txId2timestep[row.txId1], txId2timestep[row.txId2]
                            ), axis=1)
        df_filtered_edges = df_edges[df_edges['time_step'] == t].drop(
            'time_step', axis=1)

        x = torch.from_numpy(df_filtered_features.loc[:, 2:].values).to(
            torch.float)

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        mapping = {'unknown': 2, '1': 1, '2': 0}
        df_filtered_classes['class'] = df_filtered_classes['class'].map(
            mapping)

        y = torch.from_numpy(df_filtered_classes['class'].values)

        mapping = {
            idx: i
            for i, idx in enumerate(df_filtered_features['txId'].values)
        }
        df_filtered_edges['txId1'] = df_filtered_edges['txId1'].map(mapping)
        df_filtered_edges['txId2'] = df_filtered_edges['txId2'].map(mapping)
        edge_index = torch.from_numpy(
            df_filtered_edges.values).t().contiguous()

        # Timestamp based split:
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = torch.from_numpy(df_filtered_features['time_step'].values)
        train_mask = (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)
        known_mask = (y != 2)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    test_mask=test_mask, known_mask=known_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 2
