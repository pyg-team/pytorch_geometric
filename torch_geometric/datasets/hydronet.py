import os
import os.path as osp
from glob import glob
from typing import Callable, List, Optional

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
                 max_num_workers: int = 8):
        self.max_num_workers = max_num_workers
        super().__init__(root, transform, pre_transform, pre_filter)

        if not hasattr(self, '_partitions'):
            self._partitions = [
                self._create_partitions(f) for f in self.raw_paths
            ]

        [p.load() for p in tqdm(self._partitions, desc="Loading")]

    @property
    def raw_file_names(self) -> List[str]:
        path = osp.join(self.raw_dir, '*.zip')
        return sorted(glob(path))

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

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return sum(len(p) for p in self._partitions)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        if not hasattr(self, '_offsets'):
            partition_sizes = torch.tensor([len(p) for p in self._partitions])
            offsets = partition_sizes.cumsum(dim=0)
            offsets = offsets.roll(shifts=1, dims=0)
            offsets[0] = 0
            self._offsets = offsets

        partition_idx = torch.nonzero((idx - self._offsets) >= 0)[-1]
        local_idx = int(idx - offsets[partition_idx])
        return self._partitions[partition_idx].get(local_idx)


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
        self.num_clusters = int(name[1:name.find('_')])
        super().__init__(root, transform, pre_transform, pre_filter, log=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name + ".zip"]

    @property
    def processed_file_names(self) -> str:
        return self.name + '.pt'

    def process(self):
        num_nodes = self.num_clusters * 3
        chunk_size = num_nodes + 2
        y = read_energy(self.raw_paths[0], chunk_size)
        z, pos = read_atoms(self.raw_paths[0], chunk_size)
        num_graphs = z.shape[0]
        data_list = []
        (y, z, pos) = (torch.from_numpy(a) for a in (y, z, pos))

        for idx in tqdm(range(num_graphs), position=self.num_clusters - 2,
                        leave=False, desc=self.name):
            data = Data(z=z[idx, :], y=y[idx], pos=pos[idx, :])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def load(self):
        self.data, self.slices = torch.load(self.processed_paths[0])
