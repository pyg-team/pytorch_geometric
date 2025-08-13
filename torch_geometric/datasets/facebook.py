from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class FacebookPagePage(InMemoryDataset):
    r"""The Facebook Page-Page network dataset introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent verified pages on Facebook and edges are mutual likes.
    It contains 22,470 nodes, 342,004 edges, 128 node features and 4 classes.

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

    url = 'https://snap.stanford.edu/data/facebook_large.zip'

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
            f'facebook_large/{x}' for x in [
                'musae_facebook_edges.csv',
                'musae_facebook_features.json',
                'musae_facebook_target.csv',
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
        target = pd.read_csv(self.raw_paths[2])
        page_type = sorted(target['page_type'].value_counts().index.tolist())
        target['y'] = target['page_type'].apply(lambda x: page_type.index(x))

        # import pdb; pdb.set_trace()  # noqa: T201
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
        y = torch.from_numpy(target['y'].values).t().to(torch.long)
        edge_index = torch.from_numpy(edges.values).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
