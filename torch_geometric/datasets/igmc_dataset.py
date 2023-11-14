import os.path as osp
from typing import Callable, Optional

import torch
from torch import Tensor

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class IGMCDataset(InMemoryDataset):
    r"""The user-item heterogeneous rating datasets :obj:`"Douban"`,
    :obj:`"Flixster"` and :obj:`"Yahoo-Music"` from the `"Inductive Matrix
    Completion Based on Graph Neural Networks"
    <https://arxiv.org/abs/1904.12058>`_ paper.

    Nodes represent users and items.
    Edges and features between users and items represent a (training) rating of
    the item given by the user.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Douban"`,
            :obj:`"Flixster"`, :obj:`"Yahoo-Music"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://github.com/muhanzhang/IGMC/raw/master/raw_data'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace('-', '_')
        assert self.name in ['flixster', 'douban', 'yahoo_music']

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'training_test_dataset.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = f'{self.url}/{self.name}/training_test_dataset.mat'
        download_url(path, self.raw_dir)

    @staticmethod
    def load_matlab_file(path_file: str, name: str) -> Tensor:
        import h5py
        import numpy as np

        db = h5py.File(path_file, 'r')
        out = torch.from_numpy(np.asarray(db[name])).to(torch.float).t()
        db.close()

        return out

    def process(self):
        data = HeteroData()

        M = self.load_matlab_file(self.raw_paths[0], 'M')

        if self.name == 'flixster':
            user_x = self.load_matlab_file(self.raw_paths[0], 'W_users')
            item_x = self.load_matlab_file(self.raw_paths[0], 'W_movies')
        elif self.name == 'douban':
            user_x = self.load_matlab_file(self.raw_paths[0], 'W_users')
            item_x = torch.eye(M.size(1))
        elif self.name == 'yahoo_music':
            user_x = torch.eye(M.size(0))
            item_x = self.load_matlab_file(self.raw_paths[0], 'W_tracks')

        data['user'].x = user_x
        data['item'].x = item_x

        train_mask = self.load_matlab_file(self.raw_paths[0], 'Otraining')
        train_mask = train_mask.to(torch.bool)

        edge_index = train_mask.nonzero().t()
        rating = M[edge_index[0], edge_index[1]]

        data['user', 'rates', 'item'].edge_index = edge_index
        data['user', 'rates', 'item'].rating = rating

        data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'rated_by', 'user'].rating = rating

        test_mask = self.load_matlab_file(self.raw_paths[0], 'Otest')
        test_mask = test_mask.to(torch.bool)

        edge_label_index = test_mask.nonzero().t()
        edge_label = M[edge_label_index[0], edge_label_index[1]]

        data['user', 'rates', 'item'].edge_label_index = edge_label_index
        data['user', 'rates', 'item'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
