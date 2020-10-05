from itertools import product
import os
import os.path as osp
import json

import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from torch_sparse import SparseTensor
from networkx.readwrite import json_graph
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops
from google_drive_downloader import GoogleDriveDownloader as gdd


class PPILarge(InMemoryDataset):
    r"""The large version of the protein-protein interaction networks from the
    `"Predicting Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
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
    """

    adj_full_id = '1Sx3w_JK5J-lrzD2sf2ZW-CqCfnDYGRaY'
    feats_id = '15kPXApOLkXhngxMcWJDDs0fUB227h8BN'
    class_map_id = '1yBiSjpcF7tuL8UDCH0_Dcwgn01R2cKda'
    role_id = '11sr8WLA4H-JYiWRnB7xo9W9yXu2-8HQG'

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super(PPILarge, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = osp.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)

        path = osp.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)

        path = osp.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)

        path = osp.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        components = list(nx.connected_components(nx.Graph(adj)))

        adj = SparseTensor.from_scipy(adj, has_value=False)
        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        data_list = []
        for component in components:
            component = sorted(list(component))
            if len(component) < 3:
                continue
            start, end = component[0], component[-1]
            row, col, _ = adj[start:end, start:end].coo()
            edge_index = torch.stack([row, col], dim=0)

            data = Data(edge_index=edge_index, x=x[start:end], y=y[start:end])

            if self.pre_filter and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list[:20]), self.processed_paths[0])
        torch.save(self.collate(data_list[20:22]), self.processed_paths[1])
        torch.save(self.collate(data_list[22:]), self.processed_paths[2])
