import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import coalesce


class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
            If set to :obj:`True`, train/validation/test splits will be
            available as masks for multiple splits with shape
            :obj:`[num_nodes, num_splits]`. (default: :obj:`True`)
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

    raw_url = 'https://snap.stanford.edu/data/wikipedia.zip'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(
        self,
        root: str,
        name: str,
        geom_gcn_preprocess: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        # preprocess url is broken
        assert not geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[List[str], str]:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return [
                f'wikipedia/{self.name}/{x}' for x in [
                    f'musae_{self.name}_edges.csv',
                    f'musae_{self.name}_features.json',
                    f'musae_{self.name}_target.csv',
                ]
            ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)

    def process(self) -> None:
        if self.geom_gcn_preprocess:
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
            for filepath in self.raw_paths[2:]:
                masks = np.load(filepath)
                train_masks += [torch.from_numpy(masks['train_mask'])]
                val_masks += [torch.from_numpy(masks['val_mask'])]
                test_masks += [torch.from_numpy(masks['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
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
