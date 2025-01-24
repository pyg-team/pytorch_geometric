import pickle
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class BAMultiShapesDataset(InMemoryDataset):
    r"""The synthetic BA-Multi-Shapes graph classification dataset for
    evaluating explainabilty algorithms, as described in the
    `"Global Explainability of GNNs via Logic Combination of Learned Concepts"
    <https://arxiv.org/abs/2210.07147>`_ paper.

    Given three atomic motifs, namely House (H), Wheel (W), and Grid (G),
    :class:`~torch_geometric.datasets.BAMultiShapesDataset` contains 1,000
    graphs where each graph is obtained by attaching the motifs to a random
    Barabasi-Albert (BA) as follows:

    * class 0: :math:`\emptyset \lor H \lor W \lor G \lor \{ H, W, G \}`

    * class 1: :math:`(H \land W) \lor (H \land G) \lor (W \land G)`

    This dataset is pre-computed from the official implementation.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform that takes in a
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in a
            :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        pre_filter: A function that takes in a
            :class:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
        force_reload: Whether to re-process the dataset.

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
    url = ('https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/'
           'datasets/BAMultiShapes/BAMultiShapes.pkl')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'BAMultiShapes.pkl'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'rb') as f:
            adjs, xs, ys = pickle.load(f)

        data_list: List[Data] = []
        for adj, x, y in zip(adjs, xs, ys):
            edge_index = torch.from_numpy(adj).nonzero().t()
            x = torch.from_numpy(np.array(x)).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
