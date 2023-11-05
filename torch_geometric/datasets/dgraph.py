import os
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.utils import index_to_mask


class DGraphFin(InMemoryDataset):
    r"""The DGraphFin networks from the
    `"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"
    <https://arxiv.org/abs/2207.03579>`_ paper.
    It is a directed, unweighted dynamic graph consisting of millions of
    nodes and edges, representing a realistic user-to-user social network
    in financial industry.
    Node represents a Finvolution user, and an edge from one
    user to another means that the user regards the other user
    as the emergency contact person. Each edge is associated with a
    timestamp ranging from 1 to 821 and a type of emergency contact
    ranging from 0 to 11.


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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 3,700,550
          - 4,300,999
          - 17
          - 2
    """

    url = "https://dgraph.xinye.com"

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return 'DGraphFin.zip'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_classes(self) -> int:
        return 2

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path = os.path.join(self.raw_dir, "dgraphfin.npz")

        with np.load(path) as loader:
            x = torch.from_numpy(loader['x']).to(torch.float)
            y = torch.from_numpy(loader['y']).to(torch.long)
            edge_index = torch.from_numpy(loader['edge_index']).to(torch.long)
            edge_type = torch.from_numpy(loader['edge_type']).to(torch.long)
            edge_time = torch.from_numpy(loader['edge_timestamp']).to(
                torch.long)
            train_nodes = torch.from_numpy(loader['train_mask']).to(torch.long)
            val_nodes = torch.from_numpy(loader['valid_mask']).to(torch.long)
            test_nodes = torch.from_numpy(loader['test_mask']).to(torch.long)

            train_mask = index_to_mask(train_nodes, size=x.size(0))
            val_mask = index_to_mask(val_nodes, size=x.size(0))
            test_mask = index_to_mask(test_nodes, size=x.size(0))
            data = Data(x=x, edge_index=edge_index.t(), edge_type=edge_type,
                        edge_time=edge_time, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
