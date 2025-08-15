import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cornell"`, :obj:`"Texas"`,
            :obj:`"Wisconsin"`).
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
        * - Cornell
          - 183
          - 298
          - 1,703
          - 5
        * - Texas
          - 183
          - 325
          - 1,703
          - 5
        * - Wisconsin
          - 251
          - 515
          - 1,703
          - 5
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'wisconsin']

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
        out = ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
        out += [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)]
        return out

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            lines = f.read().split('\n')[1:-1]
            xs = [[float(value) for value in line.split('\t')[1].split(',')]
                  for line in lines]
            x = torch.tensor(xs, dtype=torch.float)

            ys = [int(line.split('\t')[2]) for line in lines]
            y = torch.tensor(ys, dtype=torch.long)

        with open(self.raw_paths[1]) as f:
            lines = f.read().split('\n')[1:-1]
            edge_indices = [[int(value) for value in line.split('\t')]
                            for line in lines]
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for path in self.raw_paths[2:]:
            tmp = np.load(path)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
