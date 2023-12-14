import os.path as osp
from typing import Callable, Optional

import fsspec
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class SuiteSparseMatrixCollection(InMemoryDataset):
    r"""A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix
    Collection <https://sparse.tamu.edu>`_ collected from a wide range of
    applications.

    Args:
        root (str): Root directory where the dataset should be saved.
        group (str): The group of the sparse matrix.
        name (str): The name of the sparse matrix.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'

    def __init__(
        self,
        root: str,
        group: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.group = group
        self.name = name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.group, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.group, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        fs.cp(self.url.format(self.group, self.name), self.raw_dir)

    def process(self) -> None:
        from scipy.io import loadmat

        with fsspec.open(self.raw_paths[0], 'rb') as f:
            mat = loadmat(f)['Problem'][0][0][2].tocsr().tocoo()

        row = torch.from_numpy(mat.row).to(torch.long)
        col = torch.from_numpy(mat.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        value = torch.from_numpy(mat.data).to(torch.float)
        edge_attr = None if torch.all(value == 1.0) else value

        size: Optional[torch.Size] = torch.Size(mat.shape)
        if mat.shape[0] == mat.shape[1]:
            size = None

        num_nodes = mat.shape[0]

        data = Data(edge_index=edge_index, edge_attr=edge_attr, size=size,
                    num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(group={self.group}, '
                f'name={self.name})')
