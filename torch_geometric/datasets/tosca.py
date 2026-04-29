import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array


class TOSCA(InMemoryDataset):
    r"""The TOSCA dataset from the `"Numerical Geometry of Non-Ridig Shapes"
    <https://www.amazon.com/Numerical-Geometry-Non-Rigid-Monographs-Computer/
    dp/0387733000>`_ book, containing 80 meshes.
    Meshes within the same category have the same triangulation and an equal
    number of vertices numbered in a compatible way.

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
        categories (list, optional): List of categories to include in the
            dataset. Can include the categories :obj:`"Cat"`, :obj:`"Centaur"`,
            :obj:`"David"`, :obj:`"Dog"`, :obj:`"Gorilla"`, :obj:`"Horse"`,
            :obj:`"Michael"`, :obj:`"Victoria"`, :obj:`"Wolf"`. If set to
            :obj:`None`, the dataset will contain all categories. (default:
            :obj:`None`)
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'http://tosca.cs.technion.ac.il/data/toscahires-asci.zip'

    categories = [
        'cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        categories = self.categories if categories is None else categories
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = categories
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['cat0.vert', 'cat0.tri']

    @property
    def processed_file_names(self) -> str:
        name = '_'.join([cat[:2] for cat in self.categories])
        return f'{name}.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        data_list = []
        for cat in self.categories:
            paths = glob.glob(osp.join(self.raw_dir, f'{cat}*.tri'))
            paths = [path[:-4] for path in paths]
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                pos = read_txt_array(f'{path}.vert')
                face = read_txt_array(f'{path}.tri', dtype=torch.long)
                face = face - face.min()  # Ensure zero-based index.
                data = Data(pos=pos, face=face.t().contiguous())
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])
