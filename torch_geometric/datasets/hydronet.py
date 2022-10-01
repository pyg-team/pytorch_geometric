import os
import os.path as osp
from glob import glob
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.data.summary import Stats, Summary


class HydroNet(InMemoryDataset):
    # url currently fails with unauthorized
    raw_url = ('https://g-83fdd0.1beed.03c0.data.globus.org/static_datasets/'
               'W3-W30_all_geoms_TTM2.1-F.zip')

    raw_url2 = ('https://drive.google.com/u/0/uc?confirm=t&'
                'id=18Y7OiZXSCTsHrQ83GCc4fyE_abbL6E_n')

    filename = 'W3-W30_all_geoms_TTM2.1-F.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 max_num_workers: int = 8, lazy_loading: bool = False):
        self.max_num_workers = max_num_workers
        self.lazy_loading = lazy_loading
        self._partitions = None
        self._is_loaded = False
        super().__init__(root, transform, pre_transform, pre_filter)
        self._load()

    @property
    def raw_file_names(self) -> List[str]:
        path = glob(osp.join(self.raw_dir, '*.zip'))
        path.sort(key=get_num_clusters)
        return path

    @property
    def processed_file_names(self) -> List[str]:
        names = [osp.basename(f) for f in self.raw_file_names]
        names = [osp.splitext(f)[0] for f in names]
        return [f + ".pt" for f in names]

    def download(self):
        file_path = download_url(self.raw_url2, self.raw_dir,
                                 filename=self.filename)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)
        folder_name, _ = osp.splitext(self.filename)
        files = glob(osp.join(self.raw_dir, folder_name, '*.zip'))

        for f in files:
            dst = osp.join(self.raw_dir, osp.basename(f))
            os.rename(f, dst)

        os.rmdir(osp.join(self.raw_dir, folder_name))

    def process(self):
        self._partitions = process_map(self._create_partitions,
                                       self.raw_file_names,
                                       max_workers=self.max_num_workers,
                                       position=0, leave=True)

    def _create_partitions(self, file):
        name = osp.basename(file)
        name, _ = osp.splitext(name)
        return Partition(self.root, name, self.transform, self.pre_transform,
                         self.pre_filter)

    def select_clusters(self, clusters: Union[int, List[int]]):
        clusters = [clusters] if isinstance(clusters, int) else clusters

        def is_valid_cluster(x):
            return isinstance(x, int) and x >= 3 and x <= 30

        if not all([is_valid_cluster(x) for x in clusters]):
            raise ValueError(
                "Selected clusters must be an integer in the range [3, 30]")

        self._is_loaded = False
        self._init_partitions()
        self._partitions = [self._partitions[c - 3] for c in clusters]
        return self

    def _init_partitions(self):
        self._partitions = [
            self._create_partitions(f) for f in self.raw_file_names
        ]

    def _load(self, force=False):
        if self._is_loaded:
            return

        if self._partitions is None:
            self._init_partitions()

        if not self.lazy_loading or force:
            [p.load() for p in self._partitions]
            self._is_loaded = True

    @property
    def _offsets(self):
        self._load(force=True)
        partition_sizes = torch.tensor([len(p) for p in self._partitions])
        offsets = partition_sizes.cumsum(dim=0)
        offsets = offsets.roll(shifts=1, dims=0)
        offsets[0] = 0
        return offsets

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return sum(len(p) for p in self._partitions)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        self._load(force=True)
        partition_idx = torch.nonzero((idx - self._offsets) >= 0)[-1]
        local_idx = int(idx - self._offsets[partition_idx])
        return self._partitions[partition_idx].get(local_idx)


def get_num_clusters(filepath):
    name = osp.basename(filepath)
    return int(name[1:name.find('_')])


def read_energy(file, chunk_size):
    def skipatoms(i: int):
        return (i - 1) % chunk_size != 0

    if chunk_size - 2 == 11 * 3:
        # Manually handle bad lines in W11
        df = pd.read_table(file, header=None, dtype="string",
                           skiprows=skipatoms)
        df = df[0].str.split().str[-1].astype(np.float32)
    else:
        df = pd.read_table(file, sep=r'\s+', names=["label", "energy"],
                           dtype=np.float32, skiprows=skipatoms,
                           usecols=['energy'], memory_map=True)

    return df.to_numpy().squeeze()


def read_atoms(file, chunk_size):
    def skipheaders(i: int):
        return i % chunk_size == 0 or (i - 1) % chunk_size == 0

    dtypes = {
        'atom': 'string',
        'x': np.float32,
        'y': np.float32,
        'z': np.float32
    }
    df = pd.read_table(file, sep=r'\s+', names=list(dtypes.keys()),
                       dtype=dtypes, skiprows=skipheaders, memory_map=True)

    z = np.ones(len(df), dtype=np.int8)
    z[(df.atom == 'O').to_numpy(dtype=np.bool_)] = 8
    pos = df.iloc[:, 1:4].to_numpy()
    num_nodes = (chunk_size - 2)
    num_graphs = z.shape[0] // num_nodes
    z.shape = (num_graphs, num_nodes)
    pos.shape = (num_graphs, num_nodes, 3)
    return (z, pos)


class Partition(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        self.num_clusters = get_num_clusters(name)
        super().__init__(root, transform, pre_transform, pre_filter, log=False)
        self.summary = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name + ".zip"]

    @property
    def processed_file_names(self) -> List[str]:
        return [self.name + '.pt', self.name.replace('all', 'summary') + '.pt']

    def process(self):
        num_nodes = self.num_clusters * 3
        chunk_size = num_nodes + 2
        y = read_energy(self.raw_paths[0], chunk_size)
        z, pos = read_atoms(self.raw_paths[0], chunk_size)
        num_graphs = z.shape[0]
        data_list, num_nodes_list, num_edges_list = [], [], []
        (y, z, pos) = (torch.from_numpy(a) for a in (y, z, pos))

        for idx in tqdm(range(num_graphs), position=self.num_clusters - 2,
                        leave=False, desc=self.name):
            data = Data(z=z[idx, :], y=y[idx], pos=pos[idx, :])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)

        torch.save(self.collate(data_list), self.processed_paths[0])
        summary = Summary(name=self.name, num_graphs=len(data_list),
                          num_nodes=Stats.from_data(num_nodes_list),
                          num_edges=Stats.from_data(num_edges_list))
        torch.save(summary, self.processed_paths[1])

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self.summary.num_graphs
