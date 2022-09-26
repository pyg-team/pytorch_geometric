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
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)

        data = []
        slices = []

        print("Loading...")
        for file in tqdm(self.processed_paths):
            d, s = torch.load(file)
            data.append(d)
            slices.append(s)

        print("done!")

        # TODO: figure out how to merge data/slices into one big object?

    @property
    def raw_file_names(self) -> List[str]:
        return sorted(glob(self.raw_dir + '/*/*.zip'))

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

    def process(self):
        process_map(self._process_file, self.raw_file_names, max_workers=16,
                    position=0, leave=True)

    def _process_file(self, file):
        # the number of 3-atom clusters is in the filename
        # extract this to determine the number of lines to read
        name = osp.basename(file)
        name, _ = osp.splitext(name)
        outfile = osp.join(self.processed_dir, name + ".pt")

        if osp.exists(outfile):
            return outfile

        num_clusters = int(name[1:name.find('_')])
        num_nodes = num_clusters * 3
        chunk_size = num_nodes + 2
        y = read_energy(file, chunk_size)
        z, pos = read_atoms(file, chunk_size)
        num_graphs = z.shape[0]
        data_list = []

        for idx in tqdm(range(num_graphs), position=num_clusters - 2,
                        leave=False, desc=name):
            data = Data(z=torch.from_numpy(z[idx, :]),
                        y=torch.from_numpy(y[idx]),
                        pos=torch.from_numpy(pos[idx, :]))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), outfile)
        return outfile


def read_energy(file, chunk_size):
    def skipatoms(i: int):
        return (i - 1) % chunk_size != 0

    if chunk_size - 2 == 11 * 3:
        # Manually handle bad lines in W11
        df = pd.read_table(file, header=None, dtype="string",
                           skiprows=skipatoms)
        df = df[0].str.split().str[-1].astype(float)
    else:
        df = pd.read_table(file, sep=r'\s+', names=["label", "energy"],
                           dtype="float", skiprows=skipatoms,
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
