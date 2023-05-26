import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
import scipy.sparse as sp
import h5py

from torch_geometric.data import InMemoryDataset, download_url, HeteroData


class Ratings(InMemoryDataset):
    r"""The user-item heterogeneous rating datasets :obj:'"Douban"', obj:'"Flixster"',
        and obj:'"Yahoo-music"' from the '"Inductive Matrix Completion Based on
        Graph Neural Networks" <https://arxiv.org/abs/1904.12058>' paper.
        Nodes represent users and items.
        Edges between users and items represent a rating of the item by the user
        with the rating specified in edge label.
        All three datasets are the 3000x3000 reduced versions of the original datasets
        with train/test splits as described in '"Geometric Matrix Completion with
        Recurrent Multi-Graph Neural Networks" <https://arxiv.org/abs/1704.06803>'
        Training, validation and test splits are specified on the edges.

        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): The name of the dataset (:obj:`"Douban"`, :obj:`"Flixster"`,
            :obj:`"Yahoo-music"`).
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.HeteroData` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.HeteroData` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #users
          - #items
          - #edges
          - #user features
          - #item features
        * - Douban
          - 3,000
          - 3,000
          - 136,891
          - 3,000
          - -
        * - Flixster
          - 3,000
          - 3,000
          - 26,173
          - 3,000
          - 3,000
        * - Yahoo-music
          - 3,000
          - 3,000
          - 5,335
          - -
          - 3,000

        """
    url = 'https://github.com/muhanzhang/IGMC/tree/master/raw_data'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'flixster',
            'douban',
            'yahoo_music'
        ]

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        download_url(f'{self.url}/{self.name}/training_test_dataset.mat', self.raw_dir)

    @staticmethod
    def load_matlab_file(path_file, name_field):
        db = h5py.File(path_file, 'r')
        ds = db[name_field]
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir = np.asarray(ds['ir'])
                jc = np.asarray(ds['jc'])
                out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            out = np.asarray(ds).astype(np.float32).T

        db.close()

        return out

    def process(self):
        data = HeteroData()

        M = self.load_matlab_file(self.raw_paths[0], 'M')
        training = self.load_matlab_file(self.raw_paths[0], 'Otraining')
        test = self.load_matlab_file(self.raw_paths[0], 'Otest')

        data['user'].num_nodes = M.shape[0]
        data['item'].num_nodes = M.shape[1]

        u_features, v_features = None, None

        if self.name == 'flixster':
            u_features = self.load_matlab_file(self.raw_paths[0], 'W_users')
            v_features = self.load_matlab_file(self.raw_paths[0], 'W_movies')
        elif self.name == 'douban':
            u_features = self.load_matlab_file(self.raw_paths[0], 'W_users')
            v_features = np.eye(M.shape[1])
        elif self.name == 'yahoo_music':
            v_features = self.load_matlab_file(self.raw_paths[0], 'W_tracks')
            u_features = np.eye(M.shape[0])

        data['user'].x = torch.from_numpy(u_features)
        data['item'].x = torch.from_numpy(v_features)

        # Pre-defined train-test split
        num_train = np.where(training)[0].shape[0]
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val

        train_pairs_nonzero = np.array([[u, v] for u, v in
                                        zip(np.where(training)[0], np.where(training)[1])])
        train_rating = M[np.where(training)]
        test_pairs_nonzero = np.array([[u, v] for u, v in
                                       zip(np.where(test)[0], np.where(test)[1])])
        test_rating = M[np.where(test)]

        rand_idx = list(range(len(train_pairs_nonzero)))
        np.random.shuffle(rand_idx)
        train_pairs_nonzero = train_pairs_nonzero[rand_idx]
        train_rating = train_rating[rand_idx]

        all_pairs_nonzero = np.concatenate([train_pairs_nonzero, test_pairs_nonzero], axis=0)
        rating = np.concatenate([train_rating, test_rating], axis=0)

        train_mask = torch.zeros(len(all_pairs_nonzero), dtype=bool)
        val_mask = torch.zeros(len(all_pairs_nonzero), dtype=bool)
        test_mask = torch.zeros(len(all_pairs_nonzero), dtype=bool)

        val_mask[0:num_val] = True
        train_mask[num_val:num_train + num_val] = True
        test_mask[num_train + num_val:] = True

        row, col = all_pairs_nonzero.transpose()
        data['user', 'item'].edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)], dim=0)
        data['user', 'item'].edge_label = torch.from_numpy(rating)
        data['user', 'item'].edge_train_mask = train_mask
        data['user', 'item'].edge_val_mask = val_mask
        data['user', 'item'].edge_test_mask = test_mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
