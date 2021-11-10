from typing import Optional, Callable, List

import os
import json
import os.path as osp
from collections import defaultdict

import torch
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)


class HGBDataset(InMemoryDataset):
    r"""A variety of heterogeneous graph benchmark datasets from the
    `"Are We Really Making Much Progress? Revisiting, Benchmarking, and
    Refining Heterogeneous Graph Neural Networks"
    <http://keg.cs.tsinghua.edu.cn/jietang/publications/
    KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.

    .. note::
        Test labels are randomly given to prevent data leakage issues.
        If you want to obtain final test performance, you will need to submit
        your model predictions to the
        `HGB leaderboard <https://www.biendata.xyz/hgb/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"ACM"`,
            :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
        transform (callable, optional): A function/transform that takes in an
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://cloud.tsinghua.edu.cn/d/2d965d2fc2ee41d09def/files/'
           '?p=%2F{}.zip&dl=1')

    names = {
        'acm': 'ACM',
        'dblp': 'DBLP',
        'freebase': 'Freebase',
        'imdb': 'IMDB',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in set(self.names.keys())
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name])
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = HeteroData()

        # node_types = {0: 'paper', 1, 'author', ...}
        # edge_types = {0: ('paper', 'cite', 'paper'), ...}
        if self.name in ['acm', 'dblp', 'imdb']:
            with open(self.raw_paths[0], 'r') as f:  # `info.dat`
                info = json.load(f)
            n_types = info['node.dat']['node type']
            n_types = {int(k): v for k, v in n_types.items()}
            e_types = info['link.dat']['link type']
            e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
            for key, (src, dst, rel) in e_types.items():
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                rel = rel if rel != dst and rel[1:] != dst else 'to'
                e_types[key] = (src, rel, dst)
            num_classes = len(info['label.dat']['node type']['0'])
        elif self.name in ['freebase']:
            with open(self.raw_paths[0], 'r') as f:  # `info.dat`
                info = f.read().split('\n')
            start = info.index('TYPE\tMEANING') + 1
            end = info[start:].index('')
            n_types = [v.split('\t\t') for v in info[start:start + end]]
            n_types = {int(k): v.lower() for k, v in n_types}

            e_types = {}
            start = info.index('LINK\tSTART\tEND\tMEANING') + 1
            end = info[start:].index('')
            for key, row in enumerate(info[start:start + end]):
                row = row.split('\t')[1:]
                src, dst, rel = [v for v in row if v != '']
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                e_types[key] = (src, rel, dst)
        else:
            raise NotImplementedError

        # Extract node information:
        mapping_dict = {}  # Maps global node indices to local ones.
        x_dict = defaultdict(list)
        num_nodes_dict = defaultdict(lambda: 0)
        with open(self.raw_paths[1], 'r') as f:  # `node.dat`
            xs = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for x in xs:
            n_id, n_type = int(x[0]), n_types[int(x[2])]
            mapping_dict[n_id] = num_nodes_dict[n_type]
            num_nodes_dict[n_type] += 1
            if len(x) >= 4:  # Extract features (in case they are given).
                x_dict[n_type].append([float(v) for v in x[3].split(',')])
        for n_type in n_types.values():
            if len(x_dict[n_type]) == 0:
                data[n_type].num_nodes = num_nodes_dict[n_type]
            else:
                data[n_type].x = torch.tensor(x_dict[n_type])

        edge_index_dict = defaultdict(list)
        edge_weight_dict = defaultdict(list)
        with open(self.raw_paths[2], 'r') as f:  # `link.dat`
            edges = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for src, dst, rel, weight in edges:
            e_type = e_types[int(rel)]
            src, dst = mapping_dict[int(src)], mapping_dict[int(dst)]
            edge_index_dict[e_type].append([src, dst])
            edge_weight_dict[e_type].append(float(weight))
        for e_type in e_types.values():
            edge_index = torch.tensor(edge_index_dict[e_type])
            edge_weight = torch.tensor(edge_weight_dict[e_type])
            data[e_type].edge_index = edge_index.t().contiguous()
            # Only add "weighted" edgel to the graph:
            if not torch.allclose(edge_weight, torch.ones_like(edge_weight)):
                data[e_type].edge_weight = edge_weight

        # Node classification:
        if self.name in ['acm', 'dblp', 'freebase', 'imdb']:
            with open(self.raw_paths[3], 'r') as f:  # `label.dat`
                train_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            with open(self.raw_paths[4], 'r') as f:  # `label.dat.test`
                test_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            for y in train_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]

                if not hasattr(data[n_type], 'y'):
                    num_nodes = data[n_type].num_nodes
                    if self.name in ['imdb']:  # multi-label
                        data[n_type].y = torch.zeros((num_nodes, num_classes))
                    else:
                        data[n_type].y = torch.full((num_nodes, ), -1).long()
                    data[n_type].train_mask = torch.zeros(num_nodes).bool()
                    data[n_type].test_mask = torch.zeros(num_nodes).bool()

                if data[n_type].y.dim() > 1:  # multi-label
                    for v in y[3].split(','):
                        data[n_type].y[n_id, int(v)] = 1
                else:
                    data[n_type].y[n_id] = int(y[3])
                data[n_type].train_mask[n_id] = True
            for y in test_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]
                data[n_type].test_mask[n_id] = True

        else:
            raise NotImplementedError

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'
