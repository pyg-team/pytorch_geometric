import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class Twitch(InMemoryDataset):
    r"""The Twitch Gamer networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent gamers on Twitch and edges are followerships between them.
    Node features represent embeddings of games played by the Twitch users.
    The task is to predict whether a user streams mature content.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"DE"`, :obj:`"EN"`,
            :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
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
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - DE
          - 9,498
          - 315,774
          - 128
          - 2
        * - EN
          - 7,126
          - 77,774
          - 128
          - 2
        * - ES
          - 4,648
          - 123,412
          - 128
          - 2
        * - FR
          - 6,551
          - 231,883
          - 128
          - 2
        * - PT
          - 1,912
          - 64,510
          - 128
          - 2
        * - RU
          - 4,385
          - 78,993
          - 128
          - 2
    """

    url = 'https://snap.stanford.edu/data/twitch.zip'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
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
            f'twitch/{self.name}/{x}' for x in [
                f'musae_{self.name}_edges.csv',
                f'musae_{self.name}_features.json',
                f'musae_{self.name}_target.csv',
            ]
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        file_path = download_url(self.url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)

    def process(self) -> None:
        import json

        import pandas as pd
        edges = pd.read_csv(self.raw_paths[0], dtype=int)
        features = json.load(open(self.raw_paths[1]))
        target = pd.read_csv(self.raw_paths[2], dtype=int)

        xs = []
        n_feats = 128
        for i in target['id'].values:
            f = [0] * n_feats
            if str(i) in features:
                n_len = len(features[str(i)])
                f = features[str(
                    i)][:n_feats] if n_len >= n_feats else features[str(
                        i)] + [0] * (n_feats - n_len)
            xs.append(f)
        x = torch.from_numpy(np.array(xs)).to(torch.float)
        y = torch.from_numpy(target.values[:, 1]).t().to(torch.long)
        edge_index = torch.from_numpy(edges.values).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
