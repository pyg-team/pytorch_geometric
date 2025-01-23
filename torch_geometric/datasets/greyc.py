import os
from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import fs


class GreycDataset(InMemoryDataset):
    r"""Implementation of five GREYC chemistry small datasets as pytorch
        geometric datasets : Alkane, Acyclic, MAO, Monoterpens and PAH. See
        `"GREYC's Chemistry dataset" <https://lucbrun.ensicaen.fr/CHEMISTRY/>`_
        for details.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The `name
            <https://lucbrun.ensicaen.fr/CHEMISTRY/>`_ of the
            dataset.
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

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - Acyclic
          - 183
          - ~8.2
          - ~14.3
          - 15
          - 148
        * - Alkane
          - 150
          - ~8.9
          - ~15.8
          - 15
          - 123
        * - MAO
          - 68
          - ~18.4
          - ~39.3
          - 15
          - 2
    """

    URL = ('http://localhost:3000/')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        if self.name not in {"acyclic", "alkane", "mao", "monoterpens", "pah"}:
            raise ValueError(f"Dataset {self.name} not found.")
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.data, self.slices = fs.torch_load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> str:
        return f"{self.name.lower()}.pth"

    def download(self) -> None:
        path = download_url(GreycDataset.URL + self.name + ".zip",
                            self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = fs.torch_load(
            os.path.join(self.raw_dir, self.name + ".pth"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        fs.torch_save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        name = self.name.capitalize()
        if self.name in ["mao", "pah"]:
            name = self.name.upper()
        return f'{name}({len(self)})'
