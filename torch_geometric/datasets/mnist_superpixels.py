import os

import torch
from torch_sparse import SparseTensor
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

    def __init__(self, root, train=True, transform=None, pre_transform=None,
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

    def process_raw_file(self, raw_file):
        xs, edge_indices, edge_slices, positions, ys = torch.load(raw_file)

        xs = xs.view(-1, 75, 1).unbind(dim=0)
        edge_split = (edge_slices[1:] - edge_slices[:-1]).tolist()
        edge_indices = edge_indices.to(torch.long).split(edge_split, dim=-1)
        positions = positions.view(-1, 75, 2).unbind(dim=0)
        ys = ys.view(-1).unbind(dim=0)
        size = torch.Size([75, 75])

        data_list = []
        for (row, col), x, pos, y in zip(edge_indices, xs, positions, ys):
            adj = SparseTensor(row=row, col=col, sparse_sizes=size)
            adj = adj.remove_diag().fill_cache_()
            data_list.append(Data(adj=adj, x=x, pos=pos, y=y.item()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    def process(self):
        train_data_list = self.process_raw_file(self.raw_paths[0])
        torch.save(self.collate(train_data_list), self.processed_paths[0])

        test_data_list = self.process_raw_file(self.raw_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
