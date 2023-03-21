# from signor.viz.graph import viz_graph
import logging
import os
import os.path as osp
import pickle
from typing import Optional, Callable, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.utils import remove_self_loops


class MyGNNBenchmarkDataset(InMemoryDataset):
    names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']

    url = 'https://pytorch-geometric.com/datasets/benchmarking-gnns'
    csl_url = 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1'

    def __init__(self, root: str, name: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names

        if self.name == 'CSL' and split != 'train':
            split = 'train'
            logging.warning(
                ('Dataset `CSL` does not provide a standardized splitting. '
                 'Instead, it is recommended to perform 5-fold cross '
                 'validation with stratifed sampling.'))

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
        name = self.name
        if name == 'CSL':
            return [
                'graphs_Kary_Deterministic_Graphs.pkl',
                'y_Kary_Deterministic_Graphs.pt'
            ]
        else:
            return [f'{name}_train.pt', f'{name}_val.pt', f'{name}_test.pt']

    @property
    def processed_file_names(self) -> List[str]:
        if self.name == 'CSL':
            return ['data.pt']
        else:
            return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        url = self.csl_url
        url = f'{self.url}/{self.name}.zip' if self.name != 'CSL' else url
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if self.name == 'CSL':
            data_list = self.process_CSL()
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            raise NotImplementedError

    def process_CSL(self) -> List[Data]:
        path = osp.join(self.raw_dir, 'graphs_Kary_Deterministic_Graphs.pkl')
        with open(path, 'rb') as f:
            adjs = pickle.load(f)

        path = osp.join(self.raw_dir, 'y_Kary_Deterministic_Graphs.pt')
        ys = torch.load(path).tolist()

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

    @property
    def num_tasks(self):
        return 10

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    def separate_data(self, seed, fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test
        assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 4."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        labels = self.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
