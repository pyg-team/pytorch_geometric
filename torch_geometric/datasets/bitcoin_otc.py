import datetime
import os
from typing import Callable, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class BitcoinOTC(InMemoryDataset):
    r"""The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ paper, consisting of 138
    who-trusts-whom networks of sequential time steps.

    Args:
        root (str): Root directory where the dataset should be saved.
        edge_window_size (int, optional): The window size for the existence of
            an edge in the graph sequence since its initial creation.
            (default: :obj:`10`)
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

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 138
          - 6,005
          - ~2,573.2
          - 0
          - 0
    """

    url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'

    def __init__(
        self,
        root: str,
        edge_window_size: int = 10,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'soc-sign-bitcoinotc.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        assert isinstance(self._data, Data)
        assert self._data.edge_index is not None
        return int(self._data.edge_index.max()) + 1

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            lines = [[x for x in line.split(',')]
                     for line in f.read().split('\n')[:-1]]

            edge_indices = [[int(line[0]), int(line[1])] for line in lines]
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_index = edge_index - edge_index.min()
            edge_index = edge_index.t().contiguous()
            num_nodes = int(edge_index.max()) + 1

            edge_attrs = [int(line[2]) for line in lines]
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

            stamps = [
                datetime.datetime.fromtimestamp(int(float(line[3])))
                for line in lines
            ]

        offset = datetime.timedelta(days=13.8)  # Results in 138 time steps.
        graph_indices, factor = [], 1
        for t in stamps:
            factor = factor if t < stamps[0] + factor * offset else factor + 1
            graph_indices.append(factor - 1)
        graph_idx = torch.tensor(graph_indices, dtype=torch.long)

        data_list = []
        for i in range(int(graph_idx.max()) + 1):
            mask = (graph_idx > (i - self.edge_window_size)) & (graph_idx <= i)
            data = Data()
            data.edge_index = edge_index[:, mask]
            data.edge_attr = edge_attr[mask]
            data.num_nodes = num_nodes
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
