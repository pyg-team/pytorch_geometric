from typing import Optional, Callable, List

import os.path as osp

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, extract_tar


class OMDB(InMemoryDataset):
    r"""The `Organic Materials Database (OMDB)
    <https://omdb.mathub.io/dataset>`__ of bulk organic crystals.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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
    """

    url = 'https://omdb.mathub.io/dataset'

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'OMDB-GAP1_v1.1.tar.gz'

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self):
        from ase.io import read

        extract_tar(self.raw_paths[0], self.raw_dir, log=False)
        materials = read(osp.join(self.raw_dir, 'structures.xyz'), index=':')
        bandgaps = np.loadtxt(osp.join(self.raw_dir, 'bandgaps.csv'))

        data_list = []
        for material, bandgap in zip(materials, bandgaps):
            pos = torch.from_numpy(material.get_positions()).to(torch.float)
            z = torch.from_numpy(material.get_atomic_numbers()).to(torch.int64)
            y = torch.tensor([float(bandgap)])
            data_list.append(Data(z=z, pos=pos, y=y))

        train_data = data_list[:10000]
        test_data = data_list[10000:]

        if self.pre_filter is not None:
            train_data = [d for d in train_data if self.pre_filter(d)]
            test_data = [d for d in test_data if self.pre_filter(d)]

        if self.pre_transform is not None:
            train_data = [self.pre_transform(d) for d in train_data]
            test_data = [self.pre_transform(d) for d in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(test_data), self.processed_paths[1])
