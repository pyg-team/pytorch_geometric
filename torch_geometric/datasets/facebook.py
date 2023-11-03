from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


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
    """

    url = 'https://graphmining.ai/datasets/ptg/facebook.npz'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'facebook.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
