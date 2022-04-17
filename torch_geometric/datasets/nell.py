import os
import os.path as osp
import shutil

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_tar
from torch_geometric.io import read_planetoid_data


class NELL(InMemoryDataset):
    r"""The NELL dataset, a knowledge graph from the
    `"Toward an Architecture for Never-Ending Language Learning"
    <https://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf>`_ paper.
    The dataset is processed as in the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.

    .. note::

        Entity nodes are described by sparse feature vectors of type
        :class:`torch_sparse.SparseTensor`, which can be either used directly,
        or can be converted via :obj:`data.x.to_dense()`,
        :obj:`data.x.to_scipy()` or :obj:`data.x.to_torch_sparse_coo_tensor()`.

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

    Stats:
        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 65,755
              - 251,550
              - 61,278
              - 186
    """

    url = 'http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.nell.0.001.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.root)
        extract_tar(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, 'nell_data'), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, 'nell.0.001')
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
