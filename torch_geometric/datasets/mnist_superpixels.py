import os

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)


class MNISTSuperpixels(InMemoryDataset):
    r"""MNIST superpixels dataset from the `"Geometric Deep Learning on
    Graphs and Manifolds Using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper, containing 70,000 graphs with
    75 nodes each.
    Every graph is labeled by one of 10 classes.

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

    url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
          'mnist_superpixels.tar.gz'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(MNISTSuperpixels, self).__init__(root, transform, pre_transform,
                                               pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            x, edge_index, edge_slice, pos, y = torch.load(raw_path)
            edge_index, y = edge_index.to(torch.long), y.to(torch.long)
            m, n = y.size(0), 75
            x, pos = x.view(m * n, 1), pos.view(m * n, 2)
            node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
            graph_slice = torch.arange(m + 1, dtype=torch.long)
            self.data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            self.slices = {
                'x': node_slice,
                'edge_index': edge_slice,
                'y': graph_slice,
                'pos': node_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)
