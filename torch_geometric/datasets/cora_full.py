import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import remove_self_loops, to_undirected


class CoraFull(InMemoryDataset):
    r"""The full Cora citation network dataset from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.

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
    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz'

    def __init__(self, root, transform=None, pre_transform=None):
        super(CoraFull, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'cora.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_paths[0]) as f:
            x = csr_matrix(
                (f['attr_data'], f['attr_indices'], f['attr_indptr']),
                f['attr_shape']).todense()
            x = torch.from_numpy(x).to(torch.float)

            adj = csr_matrix(
                (f['adj_data'], f['adj_indices'], f['adj_indptr']),
                f['adj_shape']).tocoo()
            edge_index = torch.tensor([adj.row, adj.col])
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index, x.size(0))

            y = torch.from_numpy(f['labels']).to(torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
