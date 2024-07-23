import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class GDELTLite(InMemoryDataset):
    r"""The (reduced) version of the Global Database of Events, Language, and
    Tone (GDELT) dataset used in the `"Do We Really Need Complicated Model
    Architectures for Temporal Networks?" <http://arxiv.org/abs/2302.11636>`_
    paper, consisting of events collected from 2016 to 2020.

    Each node (actor) holds a 413-dimensional multi-hot feature vector that
    represents CAMEO codes attached to the corresponding actor to server.

    Each edge (event) holds a timestamp and a 186-dimensional multi-hot vector
    representing CAMEO codes attached to the corresponding event to server.

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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 8,831
          - 1,912,909
          - 413
          -
    """
    url = 'https://data.pyg.org/datasets/gdelt_lite.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['node_features.pt', 'edges.csv', 'edge_features.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        x = torch.load(self.raw_paths[0])
        df = pd.read_csv(self.raw_paths[1])
        edge_attr = torch.load(self.raw_paths[2])

        row = torch.from_numpy(df['src'].values)
        col = torch.from_numpy(df['dst'].values)
        edge_index = torch.stack([row, col], dim=0)
        time = torch.from_numpy(df['time'].values).to(torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, time=time)
        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
