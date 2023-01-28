import pickle
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class BAMultiShapesDataset(InMemoryDataset):
    r"""The synthetic BAMultiShapes graph classification dataset for evaluating
    explainabilty algorithms, as described in the `"Global Explainability of GNNs
    via Logic Combination of Learned Concepts" <https://arxiv.org/abs/2210.07147>`_ paper.
    Given three atomic motifs, namely House (H), Wheel (W), and Grid (G),
    :class:`~torch_geometric.datasets.BAMultiShapesDataset` contains 1000
    graphs where each graph is obtained by attaching to a random Barabasi-Albert (BA)
    the motifs as follows:

        class 0: :math:`\emptyset` :math:`\lor` H :math:`\lor` W :math:`\lor` G :math:`\lor` :math:`\{H, W, G\}`
        class 1: H :math:`\land` W :math:`\lor` H :math:`\land` G :math:`\lor` W :math:`\land` G

    This dataset is pre-computed from the official implementation.


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
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 1000
          - 40
          - ~87.0
          - 10
          - 2
    """
    url = 'https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/datasets/BAMultiShapes'
    filename = 'BAMultiShapes.pkl'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.filename}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, x, y = pickle.load(f)

        adj = np.array(adj)
        x = np.array(x)

        adjs = torch.tensor(adj)
        xs = torch.from_numpy(x).to(torch.float)
        ys = torch.tensor(y)

        data_list: List[Data] = []
        for i in range(xs.size(0)):
            edge_index = adjs[i].nonzero().t()
            x = xs[i]
            y = int(ys[i])

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
