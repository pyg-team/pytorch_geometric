import copy
import os
import os.path as osp
from functools import cached_property
from glob import glob
from typing import Callable, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class HydroNet(InMemoryDataset):
    r"""The HydroNet dataest from the `"HydroNet: Benchmark Tasks for
    Preserving Intermolecular Interactions and Structural Motifs in Predictive
    and Generative Models for Molecular Data"
    <https://arxiv.org/abs/2012.00131>` _ paper, consisting of 5 million water
    clusters held together by hydrogen bonding networks.  This dataset
    provides atomic coordinates and total energy in kcal/mol for the cluster.

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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        max_num_workers (int): Number of multiprocessing workers to use for
            pre-processing the dataset. (default :obj:`8`)
        clusters (int or List[int], optional): Select a subset of clusters
            from the full dataset. (default :obj:`None` to select all)
    """
    # url currently fails with unauthorized
    raw_url = ('https://g-83fdd0.1beed.03c0.data.globus.org/static_datasets/'
               'W3-W30_all_geoms_TTM2.1-F.zip')

    raw_url2 = ('https://drive.google.com/u/0/uc?confirm=t&'
                'id=18Y7OiZXSCTsHrQ83GCc4fyE_abbL6E_n')

    filename = 'W3-W30_all_geoms_TTM2.1-F.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 max_num_workers: int = 8,
                 clusters: Optional[Union[int, List[int]]] = None):
        # the base Dataset class is responsible for checking if the on-disk
        # cached data is available or needs to be processed
        self.max_num_workers = max_num_workers
        if pre_filter is not None:
            warn("Ignoring the `pre_filter` argument as it is not supported "
                 "by the HydroNet dataset")

        super().__init__(root, transform, pre_transform, pre_filter)
        self.select_clusters(clusters)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'W{c}_geoms_all.zip' for c in range(3, 31)]

    @property
    def processed_file_names(self) -> List[str]:
        names = [osp.basename(f) for f in self.raw_file_names]
        names = [osp.splitext(f)[0] for f in names]
        return [f + ".npz" for f in names]

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
        from tqdm.contrib.concurrent import process_map

        self._partitions = process_map(self._create_partitions, self.raw_paths,
                                       max_workers=self.max_num_workers,
                                       position=0, leave=True)

    def _create_partitions(self, file):
        name = osp.basename(file)
        name, _ = osp.splitext(name)
        return Partition(self.root, name, self.transform, self.pre_transform)

    def select_clusters(self, clusters: Union[int, List[int]]):
        self._partitions = [self._create_partitions(f) for f in self.raw_paths]

        if clusters is None:
            return

        clusters = [clusters] if isinstance(clusters, int) else clusters

        def is_valid_cluster(x):
            return isinstance(x, int) and x >= 3 and x <= 30

        if not all([is_valid_cluster(x) for x in clusters]):
            raise ValueError(
                "Selected clusters must be an integer in the range [3, 30]")

        self._partitions = [self._partitions[c - 3] for c in clusters]
        return self

    def pre_load(self):
        [p.load() for p in tqdm(self._partitions)]

    @cached_property
    def _dataset(self):
        return ConcatDataset(self._partitions)

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return sum(len(p) for p in self._partitions)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        return self._dataset[idx]


def get_num_clusters(filepath):
    name = osp.basename(filepath)
    return int(name[1:name.find('_')])


def read_energy(file, chunk_size):
    import pandas as pd

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
    import pandas as pd

    def skipheaders(i: int):
        return i % chunk_size == 0 or (i - 1) % chunk_size == 0

    dtypes = {
        'atom': 'string',
        'x': np.float16,
        'y': np.float16,
        'z': np.float16
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
                 backend: Optional[str] = 'npz'):
        self.name = name
        self.num_clusters = get_num_clusters(name)
        self.backend = backend
        super().__init__(root, transform, pre_transform, pre_filter=None,
                         log=False)
        self.is_loaded = False

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name + ".zip"]

    @property
    def processed_file_names(self) -> List[str]:
        return [self.name + '.npz']

    def process(self):
        num_nodes = self.num_clusters * 3
        chunk_size = num_nodes + 2
        z, pos = read_atoms(self.raw_paths[0], chunk_size)
        y = read_energy(self.raw_paths[0], chunk_size)
        np.savez_compressed(self.processed_paths[0], z=z, pos=pos, y=y,
                            num_graphs=z.shape[0])

    def load(self):
        if self.is_loaded:
            return

        with np.load(self.processed_paths[0]) as npzfile:
            self.z = npzfile['z']
            self.pos = npzfile['pos']
            self.y = npzfile['y']

        self.is_loaded = True

    @cached_property
    def num_graphs(self) -> int:
        with np.load(self.processed_paths[0]) as npzfile:
            return int(npzfile['num_graphs'])

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self.num_graphs

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        self.load()
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = Data(z=torch.from_numpy(self.z[idx, :]),
                    pos=torch.from_numpy(self.pos[idx, :, :]),
                    y=float(self.y[idx]))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self._data_list[idx] = copy.copy(data)
        return data
