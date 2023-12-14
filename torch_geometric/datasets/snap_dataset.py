import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Union

import fsspec
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.utils import coalesce


class EgoData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == 'circle':
            return self.num_nodes
        elif key == 'circle_batch':
            return int(value.max()) + 1 if value.numel() > 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def read_ego(files: List[str], name: str) -> List[EgoData]:
    import pandas as pd

    all_featnames = []
    files = [
        x for x in files if x.split('.')[-1] in
        ['circles', 'edges', 'egofeat', 'feat', 'featnames']
    ]
    for i in range(4, len(files), 5):
        featnames_file = files[i]
        with fsspec.open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    all_featnames_dict = {key: i for i, key in enumerate(all_featnames)}

    data_list = []
    for i in range(0, len(files), 5):
        circles_file = files[i]
        edges_file = files[i + 1]
        egofeat_file = files[i + 2]
        feat_file = files[i + 3]
        featnames_file = files[i + 4]

        x = None
        if name != 'gplus':  # Don't read node features on g-plus:
            x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
            x_ego = torch.from_numpy(x_ego.values)

            x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
            x = torch.from_numpy(x.values)[:, 1:]

            x_all = torch.cat([x, x_ego], dim=0)

            # Reorder `x` according to `featnames` ordering.
            x_all = torch.zeros(x.size(0), len(all_featnames))
            with fsspec.open(featnames_file, 'r') as f:
                featnames = f.read().split('\n')[:-1]
                featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            indices = [all_featnames_dict[featname] for featname in featnames]
            x_all[:, torch.tensor(indices)] = x
            x = x_all

        idx = pd.read_csv(feat_file, sep=' ', header=None, dtype=str,
                          usecols=[0]).squeeze()

        idx_assoc: Dict[str, int] = {}
        for i, j in enumerate(idx):
            idx_assoc[j] = i

        circles: List[int] = []
        circles_batch: List[int] = []
        with fsspec.open(circles_file, 'r') as f:
            for i, line in enumerate(f.read().split('\n')[:-1]):
                circle_indices = [idx_assoc[c] for c in line.split()[1:]]
                circles += circle_indices
                circles_batch += [i] * len(circle_indices)
        circle = torch.tensor(circles)
        circle_batch = torch.tensor(circles_batch)

        try:
            row = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[0]).squeeze()
            col = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[1]).squeeze()
        except Exception:
            continue

        row = torch.tensor([idx_assoc[i] for i in row])
        col = torch.tensor([idx_assoc[i] for i in col])

        N = max(int(row.max()), int(col.max())) + 2
        N = x.size(0) if x is not None else N

        row_ego = torch.full((N - 1, ), N - 1, dtype=torch.long)
        col_ego = torch.arange(N - 1)

        # Ego node should be connected to every other node.
        row = torch.cat([row, row_ego, col_ego], dim=0)
        col = torch.cat([col, col_ego, row_ego], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = coalesce(edge_index, num_nodes=N)

        data = EgoData(x=x, edge_index=edge_index, circle=circle,
                       circle_batch=circle_batch)

        data_list.append(data)

    return data_list


def read_soc(files: List[str], name: str) -> List[Data]:
    import pandas as pd

    skiprows = 4
    if name == 'pokec':
        skiprows = 0

    edge_index = pd.read_csv(files[0], sep='\t', header=None,
                             skiprows=skiprows, dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()
    num_nodes = edge_index.max().item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


def read_wiki(files: List[str], name: str) -> List[Data]:
    import pandas as pd

    edge_index = pd.read_csv(files[0], sep='\t', header=None, skiprows=4,
                             dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()

    idx = torch.unique(edge_index.flatten())
    idx_assoc = torch.full((edge_index.max() + 1, ), -1, dtype=torch.long)
    idx_assoc[idx] = torch.arange(idx.size(0))

    edge_index = idx_assoc[edge_index]
    num_nodes = edge_index.max().item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


class SNAPDataset(InMemoryDataset):
    r"""A variety of graph datasets collected from `SNAP at Stanford University
    <https://snap.stanford.edu/data>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset.
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'ego-facebook': ['facebook.tar.gz'],
        'ego-gplus': ['gplus.tar.gz'],
        'ego-twitter': ['twitter.tar.gz'],
        'soc-ca-astroph': ['ca-AstroPh.txt.gz'],
        'soc-ca-grqc': ['ca-GrQc.txt.gz'],
        'soc-epinions1': ['soc-Epinions1.txt.gz'],
        'soc-livejournal1': ['soc-LiveJournal1.txt.gz'],
        'soc-pokec': ['soc-pokec-relationships.txt.gz'],
        'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'],
        'soc-slashdot0922': ['soc-Slashdot0902.txt.gz'],
        'wiki-vote': ['wiki-Vote.txt.gz'],
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.available_datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def _download(self) -> None:
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def download(self) -> None:
        for name in self.available_datasets[self.name]:
            fs.cp(f'{self.url}/{name}', self.raw_dir, extract=True)

    def process(self) -> None:
        raw_dir = self.raw_dir
        filenames = fs.ls(self.raw_dir)
        if len(filenames) == 1 and fs.isdir(filenames[0]):
            raw_dir = filenames[0]

        raw_files = fs.ls(raw_dir)

        data_list: Union[List[Data], List[EgoData]]
        if self.name[:4] == 'ego-':
            data_list = read_ego(raw_files, self.name[4:])
        elif self.name[:4] == 'soc-':
            data_list = read_soc(raw_files, self.name[:4])
        elif self.name[:5] == 'wiki-':
            data_list = read_wiki(raw_files, self.name[5:])
        else:
            raise NotImplementedError

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'SNAP-{self.name}({len(self)})'
