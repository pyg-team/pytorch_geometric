import os.path as osp
from glob import glob
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply


class CoMA(InMemoryDataset):
    r"""The CoMA 3D faces dataset from the `"Generating 3D faces using
    Convolutional Mesh Autoencoders" <https://arxiv.org/abs/1807.10267>`_
    paper, containing 20,466 meshes of extreme expressions captured over 12
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
        root (str): Root directory where the dataset should be saved.
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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 20,465
          - 5,023
          - 29,990
          - 3
          - 12
    """

    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'COMA_data.zip'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download 'COMA_data.zip' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self):
        folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))
        if len(folders) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))

        train_data_list, test_data_list = [], []
        for folder in folders:
            for i, category in enumerate(self.categories):
                files = sorted(glob(osp.join(folder, category, '*.ply')))
                for j, f in enumerate(files):
                    data = read_ply(f)
                    data.y = torch.tensor([i], dtype=torch.long)
                    if self.pre_filter is not None and\
                       not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if (j % 100) < 90:
                        train_data_list.append(data)
                    else:
                        test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
