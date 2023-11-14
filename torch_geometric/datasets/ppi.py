import json
import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops


class PPI(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 20
          - ~2,245.3
          - ~61,318.4
          - 50
          - 121
    """

    url = 'https://data.dgl.ai/dataset/ppi.zip'

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):

        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if split == 'train':
            self.load(self.processed_paths[0])
        elif split == 'val':
            self.load(self.processed_paths[1])
        elif split == 'test':
            self.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> List[str]:
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return [f'{split}_{name}' for split, name in product(splits, files)]

    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        import networkx as nx
        from networkx.readwrite import json_graph

        for s, split in enumerate(['train', 'valid', 'test']):
            path = osp.join(self.raw_dir, f'{split}_graph.json')
            with open(path, 'r') as f:
                G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

            x = np.load(osp.join(self.raw_dir, f'{split}_feats.npy'))
            x = torch.from_numpy(x).to(torch.float)

            y = np.load(osp.join(self.raw_dir, f'{split}_labels.npy'))
            y = torch.from_numpy(y).to(torch.float)

            data_list = []
            path = osp.join(self.raw_dir, f'{split}_graph_id.npy')
            idx = torch.from_numpy(np.load(path)).to(torch.long)
            idx = idx - idx.min()

            for i in range(idx.max().item() + 1):
                mask = idx == i

                G_s = G.subgraph(
                    mask.nonzero(as_tuple=False).view(-1).tolist())
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index = edge_index - edge_index.min()
                edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, x=x[mask], y=y[mask])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            self.save(data_list, self.processed_paths[s])
