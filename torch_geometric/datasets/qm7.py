from typing import Callable, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class QM7b(InMemoryDataset):
    r"""The QM7b dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    7,211 molecules with 14 regression targets.

    Args:
        root (str): Root directory where the dataset should be saved.
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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 7,211
          - ~15.4
          - ~245.0
          - 0
          - 14
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'qm7b.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        from scipy.io import loadmat

        data = loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])
        target = torch.from_numpy(data['T']).to(torch.float)

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y = target[i].view(1, -1)
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
