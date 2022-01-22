import logging
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops


class GNNBenchmarkDataset(InMemoryDataset):
    r"""A variety of artificially and semi-artificially generated graph
    datasets from the `"Benchmarking Graph Neural Networks"
    <https://arxiv.org/abs/2003.00982>`_ paper.

    .. note::
        The ZINC dataset is provided via
        :class:`torch_geometric.datasets.ZINC`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - PATTERN
              - 10,000
              - ~118.9
              - ~6,098.9
              - 3
              - 2
            * - CLUSTER
              - 10,000
              - ~117.2
              - ~4,303.9
              - 7
              - 6
            * - MNIST
              - 55,000
              - ~70.6
              - ~564.5
              - 3
              - 10
            * - CIFAR10
              - 45,000
              - ~117.6
              - ~941.2
              - 5
              - 10
            * - TSP
              - 10,000
              - ~275.4
              - ~6,885.0
              - 2
              - 2
            * - CSL
              - 150
              - ~41.0
              - ~164.0
              - 0
              - 10
    """

    names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']

    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    urls = {
        'PATTERN': f'{root_url}/PATTERN_v2.zip',
        'CLUSTER': f'{root_url}/CLUSTER_v2.zip',
        'MNIST': f'{root_url}/MNIST_v2.zip',
        'CIFAR10': f'{root_url}/CIFAR10_v2.zip',
        'TSP': f'{root_url}/TSP_v2.zip',
        'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1',
    }

    def __init__(self, root: str, name: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names

        if self.name == 'CSL' and split != 'train':
            split = 'train'
            logging.warning(
                ("Dataset 'CSL' does not provide a standardized splitting. "
                 "Instead, it is recommended to perform 5-fold cross "
                 "validation with stratifed sampling"))

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")

        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == 'CSL':
            return [
                'graphs_Kary_Deterministic_Graphs.pkl',
                'y_Kary_Deterministic_Graphs.pt'
            ]
        else:
            name = self.urls[self.name].split('/')[-1][:-4]
            return [f'{name}.pt']

    @property
    def processed_file_names(self) -> List[str]:
        if self.name == 'CSL':
            return ['data.pt']
        else:
            return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if self.name == 'CSL':
            data_list = self.process_CSL()
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            inputs = torch.load(self.raw_paths[0])
            for i in range(len(inputs)):
                data_list = [Data(**data_dict) for data_dict in inputs[i]]

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in data_list]

                torch.save(self.collate(data_list), self.processed_paths[i])

    def process_CSL(self) -> List[Data]:
        with open(self.raw_paths[0], 'rb') as f:
            adjs = pickle.load(f)

        ys = torch.load(self.raw_paths[1]).tolist()

        data_list = []
        for adj, y in zip(adjs, ys):
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            data = Data(edge_index=edge_index, y=y, num_nodes=adj.shape[0])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
