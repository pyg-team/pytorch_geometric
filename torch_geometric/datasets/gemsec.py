import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class GemsecDeezer(InMemoryDataset):
    r"""The Deezer User Network datasets introduced in the
    `"GEMSEC: Graph Embedding with Self Clustering"
    <https://arxiv.org/abs/1802.03997>`_ paper.
    Nodes represent Deezer user and edges are mutual friendships.
    The task is multi-label multi-class node classification about
    the genres liked by the users.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"HU"`, :obj:`"HR"`,
            :obj:`"RO"`).
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

    url = 'https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in ['HU', 'HR', 'RO']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'deezer_clean_data/{x}' for x in [
                f'{self.name}_edges.csv',
                f'{self.name}_genres.json',
            ]
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        file_path = download_url(self.url, self.raw_dir)
        extract_tar(file_path, self.raw_dir)

    def process(self) -> None:

        import pandas as pd
        edges = pd.read_csv(self.raw_paths[0], dtype=int)

        edge_index = torch.from_numpy(edges.values).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
