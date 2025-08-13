from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class LastFMAsia(InMemoryDataset):
    r"""The LastFM Asia Network dataset introduced in the `"Characteristic
    Functions on Graphs: Birds of a Feather, from Statistical Descriptors to
    Parametric Models" <https://arxiv.org/abs/2005.07959>`_ paper.
    Nodes represent LastFM users from Asia and edges are friendships.
    It contains 7,624 nodes, 55,612 edges, 128 node features and 18 classes.

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
    """

    url = 'https://snap.stanford.edu/data/lastfm_asia.zip'

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
    def raw_file_names(self) -> str:
        return [
            f'lasftm_asia/{x}' for x in [
                'lastfm_asia_edges.csv',
                'lastfm_asia_features.json',
                'lastfm_asia_target.csv',
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

        x = []
        n_feats = 128
        for i in target['id'].values:
            f = [0] * n_feats
            if str(i) in features:
                n_len = len(features[str(i)])
                f = features[str(
                    i)][:n_feats] if n_len >= n_feats else features[str(
                        i)] + [0] * (n_feats - n_len)
            x.append(f)
        x = torch.from_numpy(np.array(x)).to(torch.float)
        y = torch.from_numpy(target.values[:, 1]).t().to(torch.long)
        edge_index = torch.from_numpy(edges.values).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
