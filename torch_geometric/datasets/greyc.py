import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class GreycDataset(InMemoryDataset):
    r"""Implementation of five GREYC chemistry small datasets as pytorch
        geometric datasets : Acyclic, Alkane, MAO, Monoterpens and PAH. See
        `"GREYC's Chemistry dataset" <https://lucbrun.ensicaen.fr/CHEMISTRY/>`_
        for details.

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
        :widths: 20 10 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
          - #type
        * - Acyclic
          - 183
          - ~8.2
          - ~14.3
          - 7
          - 148
          - Regression
        * - Alkane
          - 149
          - ~8.9
          - ~15.7
          - 4
          - 123
          - Regression
        * - MAO
          - 68
          - ~18.4
          - ~39.3
          - 7
          - 2
          - Classification
        * - Monoterpens
          - 302
          - ~11.0
          - ~22.2
          - 7
          - 10
          - Classification
        * - PAH
          - 94
          - ~20.7
          - ~48.9
          - 4
          - 2
          - Classification
    """

    URL = ('https://raw.githubusercontent.com/bgauzere/'
           'greycdata/refs/heads/main/greycdata/data_gml/')

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
        self.load(self.processed_paths[0])

    def __repr__(self) -> str:
        name = self.name.capitalize()
        if self.name in ["mao", "pah"]:
            name = self.name.upper()
        return f'{name}({len(self)})'

    @staticmethod
    def _gml_to_data(gml: str, gml_file: bool = True) -> Data:
        """Reads a `gml` file and creates a `Data` object.

        Parameters :
        ------------

        * `gml`      : gml file path if `gml_file` is
                       set to `True`, gml content otherwise.
        * `gml_file` : indicates whether `gml` is a path
                       to a gml file or a gml file content.

        Returns :
        ---------

        * `Data` : the `Data` object created from the file content.

        Raises :
        --------

        * `FileNotFoundError` : if `gml` is a file path and doesn't exist.
        """
        import networkx as nx
        if gml_file:
            if not os.path.exists(gml):
                raise FileNotFoundError(f"File `{gml}` does not exist")
            g: nx.Graph = nx.read_gml(gml)
        else:
            g: nx.Graph = nx.parse_gml(gml)

        x, edge_index, edge_attr = [], [], []

        y = g.graph["y"] if "y" in g.graph else None
        dtype = torch.float if isinstance(y, float) else torch.long
        y = torch.tensor([y], dtype=dtype) if y is not None else None

        for _, attr in g.nodes(data=True):
            x.append(attr["x"])

        for u, v, attr in g.edges(data=True):
            edge_index.append([int(u), int(v)])
            edge_index.append([int(v), int(u)])
            edge_attr.append(attr["edge_attr"])
            edge_attr.append(attr["edge_attr"])

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index,
                                  dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)

    @staticmethod
    def _load_gml_data(gml: str) -> List[Data]:
        """Reads a dataset from a gml file
        and converts it into a list of `Data`.

        Parameters :
        ------------

        * `gml` : Source filename. If `zip` file, it will be extracted.

        Returns :
        ---------

        * `List[Data]` : Content of the gml dataset.
        """
        GML_SEPARATOR = "---"
        with open(gml, encoding="utf8") as f:
            gml_contents = f.read()
        gml_files = gml_contents.split(GML_SEPARATOR)
        return [
            GreycDataset._gml_to_data(content, False) for content in gml_files
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.gml"

    @property
    def num_node_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_edge_features(self) -> int:
        return int(self.edge_attr.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.y.unique())

    def download(self) -> None:
        path = download_url(GreycDataset.URL + self.name + ".zip",
                            self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = self._load_gml_data(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
