import os.path as osp
import random

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split (:obj:`"public"`,
            :obj:`"full"`, :obj:`"random"`).  If set to :obj:`"public"`, the
            split will be the public fixed split from the `"Revisiting
            Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper. If set to :obj:`"full"`,
            all samples except validation and test sets in :obj:`"public"` split
            will be used for training, from `"FastGCN: Fast Learning with
            Graph Convolutional Networks via Importance Sampling"
            <https://arxiv.org/abs/1801.10247>`_ paper. If set to :obj:`"random"`,
            train, validation, and test sets will be randomly sampled.
            (default: :obj:`"public"`)
        num_train_per_class (int, optional): number of training samples per
            class, for :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): number of validation samples per for
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): number of test samples per for
            :obj:`"random"` split. (default: :obj:`1000`)
        seed (int, optional): seed for :obj:`"random"`
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, split="public",
                 num_train_per_class=20, num_val=500, num_test=1000, seed=42,
                 transform=None, pre_transform=None):
        self.name = name

        self.split = split
        assert self.split in ["public", "full", "random"]

        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed

        super(Planetoid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed', self.split)

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == "full":
            data.train_mask[:] = True
            data.train_mask[data.val_mask] = False
            data.train_mask[data.test_mask] = False

        elif self.split == "random":
            data.train_mask[:] = False
            data.val_mask[:] = False
            data.test_mask[:] = False

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def _get_data_with_random_train_val_test_mask(self, data):
        random.seed(self.seed)
        for c in range(self.num_classes):
            idx = random.sample((data.y == c).nonzero().squeeze().tolist(), self.num_train_per_class)
            data.train_mask[torch.Tensor(idx).long()] = True
        val_test_idx = random.sample((~data.train_mask).nonzero().squeeze().tolist(),
                                     self.num_val + self.num_test)
        data.val_mask[val_test_idx[:self.num_val]] = True
        data.test_mask[val_test_idx[self.num_val:]] = True
        return data

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.split == "random":
            data = self._get_data_with_random_train_val_test_mask(data)
        return data

    def __repr__(self):
        return '{}()'.format(self.name)
