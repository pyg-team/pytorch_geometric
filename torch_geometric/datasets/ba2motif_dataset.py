import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class BA2MotifDataset(InMemoryDataset):
    r"""The synthetic BA-2motifs graph classification dataset for evaluating
    explainabilty algorithms, as described in the `"Parameterized Explainer
    for Graph Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.
    :class:`~torch_geometric.datasets.BA2MotifDataset` contains 1000 random
    Barabasi-Albert (BA) graphs.
    Half of the graphs are attached with a
    :class:`~torch_geometric.datasets.motif_generator.HouseMotif`, and the rest
    are attached with a five-node
    :class:`~torch_geometric.datasets.motif_generator.CycleMotif`.
    The graphs are assigned to one of the two classes according to the type of
    attached motifs.

    This dataset is pre-computed from the official implementation. If you want
    to create own variations of it, you can make use of the
    :class:`~torch_geometric.datasets.ExplainerDataset`:

    .. code-block:: python

        import torch
        from torch_geometric.datasets import ExplainerDataset
        from torch_geometric.datasets.graph_generator import BAGraph
        from torch_geometric.datasets.motif_generator import HouseMotif
        from torch_geometric.datasets.motif_generator import CycleMotif

        dataset1 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=HouseMotif(),
            num_motifs=1,
            num_graphs=500,
        )

        dataset2 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=CycleMotif(5),
            num_motifs=1,
            num_graphs=500,
        )

        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

    Args:
        root (string): Root directory where the dataset should be saved.
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
          - 25
          - ~51.0
          - 10
          - 2
    """
    url = 'https://github.com/flyingdoog/PGExplainer/raw/master/dataset'
    filename = 'BA-2motif.pkl'

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

        adjs = torch.from_numpy(adj)
        xs = torch.from_numpy(x).to(torch.float)
        ys = torch.from_numpy(y)

        data_list: List[Data] = []
        for i in range(xs.size(0)):
            edge_index = adjs[i].nonzero().t()
            x = xs[i]
            y = int(ys[i].nonzero())

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
