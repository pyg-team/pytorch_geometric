import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class AQSOL(InMemoryDataset):
    r"""The AQSOL dataset from the `Benchmarking Graph Neural Networks
    <http://arxiv.org/abs/2003.00982>`_ paper based on
    `AqSolDB <https://www.nature.com/articles/s41597-019-0151-1>`_, a
    standardized database of 9,982 molecular graphs with their aqueous
    solubility values, collected from 9 different data sources.

    The aqueous solubility targets are collected from experimental measurements
    and standardized to LogS units in AqSolDB. These final values denote the
    property to regress in the :class:`AQSOL` dataset. After filtering out few
    graphs with no bonds/edges, the total number of molecular graphs is 9,833.
    For each molecular graph, the node features are the types of heavy atoms
    and the edge features are the types of bonds between them, similar as in
    the :class:`~torch_geometric.datasets.ZINC` dataset.

    Args:
        root (string): Root directory where the dataset should be saved.
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
            value, indicating whether the data object should be included in
            the final dataset. (default: :obj:`None`)
    """
    url = 'https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1'

    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'atom_dict.pickle',
            'bond_dict.pickle'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'asqol_graph_raw'), self.raw_dir)
        os.unlink(path)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graphs = pickle.load(f)

            data_list: List[Data] = []
            for graph in graphs:
                x, edge_attr, edge_index, y = graph

                x = torch.from_numpy(x)
                edge_attr = torch.from_numpy(edge_attr)
                edge_index = torch.from_numpy(edge_index)
                y = torch.tensor([y]).float()

                if edge_index.numel() == 0:
                    continue  # Skipping for graphs with no bonds/edges.

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), path)

    def atoms(self) -> List[str]:
        return [
            'Br', 'C', 'N', 'O', 'Cl', 'Zn', 'F', 'P', 'S', 'Na', 'Al', 'Si',
            'Mo', 'Ca', 'W', 'Pb', 'B', 'V', 'Co', 'Mg', 'Bi', 'Fe', 'Ba', 'K',
            'Ti', 'Sn', 'Cd', 'I', 'Re', 'Sr', 'H', 'Cu', 'Ni', 'Lu', 'Pr',
            'Te', 'Ce', 'Nd', 'Gd', 'Zr', 'Mn', 'As', 'Hg', 'Sb', 'Cr', 'Se',
            'La', 'Dy', 'Y', 'Pd', 'Ag', 'In', 'Li', 'Rh', 'Nb', 'Hf', 'Cs',
            'Ru', 'Au', 'Sm', 'Ta', 'Pt', 'Ir', 'Be', 'Ge'
        ]

    def bonds(self) -> List[str]:
        return ['NONE', 'SINGLE', 'DOUBLE', 'AROMATIC', 'TRIPLE']
