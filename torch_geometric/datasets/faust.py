import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply


class FAUST(InMemoryDataset):
    r"""The FAUST humans dataset from the `"FAUST: Dataset and Evaluation for
    3D Mesh Registration"
    <http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>`_ paper,
    containing 100 watertight meshes representing 10 different poses for 10
    different subjects.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

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

    url = 'http://faust.is.tue.mpg.de/'

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'MPI-FAUST.zip'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        path = osp.join(path, 'tr_reg_{0:03d}.ply')
        data_list = []
        for i in range(100):
            data = read_ply(path.format(i))
            data.y = torch.tensor([i % 10], dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])

        shutil.rmtree(osp.join(self.raw_dir, 'MPI-FAUST'))
