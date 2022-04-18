import logging
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import download_url, extract_tar, extract_zip


class Entities(InMemoryDataset):
    r"""The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from
    the `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"AIFB"`,
            :obj:`"MUTAG"`, :obj:`"BGS"`, :obj:`"AM"`).
        hetero (bool, optional): If set to :obj:`True`, will save the dataset
            as a :class:`~torch_geometric.data.HeteroData` object.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # url = 'https://data.dgl.ai/dataset/{}.tgz'

    def __init__(self, root: str, name: str, split='train',
                 hetero: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.hetero = hetero
        assert self.name in ['aifb', 'am', 'mutag', 'bgs', 'ppi']
        if self.name == 'ppi':
            self.url = 'https://www.dropbox.com/s/fzxpoxb1k8m9jrg/ppi.zip?dl=1'
        else:
            self.url = 'https://data.dgl.ai/dataset/{}.tgz'


        super().__init__(root, transform, pre_transform)
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def num_relations(self) -> int:
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> int:
        if self.name=='ppi':
            return int(self.data.y.max().item() + 1)
        else:
            return self.data.train_y.max().item() + 1

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == 'ppi':
            from itertools import product
            splits = ['train', 'valid', 'test']
            files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
            return ['{}_{}'.format(s, f) for s, f in product(splits, files)]
        else:
            return [
                f'{self.name}_stripped.nt.gz',
                'completeDataset.tsv',
                'trainingSet.tsv',
                'testSet.tsv',
            ]

    @property
    def processed_file_names(self) -> str:
        if self.name == 'ppi':
            return ['train.pt', 'val.pt', 'test.pt']
        else:
            return 'hetero_data.pt' if self.hetero else 'data.pt'

    def download(self):
        if self.name == 'ppi':
            path = download_url(self.url.format(self.name), self.root)
            extract_zip(path, self.raw_dir)
        else:
            path = download_url(self.url.format(self.name), self.root)
            extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if self.name == 'ppi':
            import networkx as nx
            import json
            from networkx.readwrite import json_graph

            for s, split in enumerate(['train', 'valid', 'test']):
                path = osp.join(self.raw_dir, '{}_graph.json').format(split)
                with open(path, 'r') as f:
                    G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))
                x = np.load(osp.join(self.raw_dir, '{}_feats.npy').format(split))
                x = torch.from_numpy(x).to(torch.float)

                y = np.load(osp.join(self.raw_dir, '{}_labels.npy').format(split))
                y1 = np.argmax(y, axis=1)
                # print(f"y={y}, y1={y1}")
                y = torch.from_numpy(y1).to(torch.long)

                data_list = []
                path = osp.join(self.raw_dir, '{}_graph_id.npy').format(split)
                idx = torch.from_numpy(np.load(path)).to(torch.long)
                idx = idx - idx.min()
                for i in range(idx.max().item() + 1):
                    mask = idx == i
                    G_s = G.subgraph(
                        mask.nonzero(as_tuple=False).view(-1).tolist())
                    edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                    edge_index = edge_index - edge_index.min()
                    edge_weight = torch.tensor(
                        [np.array(list(format(data['weight'], '03b'))).astype("float")
                         for _, _, data in G_s.edges(data=True)], dtype=torch.int64)
                    edge_index_t = torch.transpose(edge_index, 0, 1).numpy()
                    new_edge_index = []
                    edge_type = []
                    for i, weight in enumerate(edge_weight):
                        for j, bit in enumerate(weight):
                            if bit:
                                new_edge_index.append(edge_index_t[i])
                                edge_type.append(j)
                                new_edge_index.append(edge_index_t[i][::-1])
                                edge_type.append(2*j+1)
                    # print(f"updated one hot weight {edge_weight}")
                    updated_edge_index = torch.tensor(
                        np.array(new_edge_index).T, dtype=torch.int64)
                    data = Data(edge_index=updated_edge_index,
                                x=x[mask], y=y[mask],
                                edge_type=torch.tensor(edge_type, dtype=torch.int64))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                torch.save(self.collate(data_list), self.processed_paths[s])
        else:

            import gzip
            import pandas as pd
            import rdflib as rdf

            graph_file, task_file, train_file, test_file = self.raw_paths

            with hide_stdout():
                g = rdf.Graph()
                with gzip.open(graph_file, 'rb') as f:
                    g.parse(file=f, format='nt')

            freq = Counter(g.predicates())

            relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
            subjects = set(g.subjects())
            objects = set(g.objects())
            nodes = list(subjects.union(objects))

            N = len(nodes)
            R = 2 * len(relations)

            relations_dict = {rel: i for i, rel in enumerate(relations)}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            edges = []
            for s, p, o in g.triples((None, None, None)):
                src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
                edges.append([src, dst, 2 * rel])
                edges.append([dst, src, 2 * rel + 1])

            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            perm = (N * R * edges[0] + R * edges[1] + edges[2]).argsort()
            edges = edges[:, perm]

            edge_index, edge_type = edges[:2], edges[2]

            if self.name == 'am':
                label_header = 'label_cateogory'
                nodes_header = 'proxy'
            elif self.name == 'aifb':
                label_header = 'label_affiliation'
                nodes_header = 'person'
            elif self.name == 'mutag':
                label_header = 'label_mutagenic'
                nodes_header = 'bond'
            elif self.name == 'bgs':
                label_header = 'label_lithogenesis'
                nodes_header = 'rock'

            labels_df = pd.read_csv(task_file, sep='\t')
            labels_set = set(labels_df[label_header].values.tolist())
            labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
            nodes_dict = {np.unicode(key): val for key, val in nodes_dict.items()}

            train_labels_df = pd.read_csv(train_file, sep='\t')
            train_indices, train_labels = [], []
            for nod, lab in zip(train_labels_df[nodes_header].values,
                                train_labels_df[label_header].values):
                train_indices.append(nodes_dict[nod])
                train_labels.append(labels_dict[lab])

            train_idx = torch.tensor(train_indices, dtype=torch.long)
            train_y = torch.tensor(train_labels, dtype=torch.long)

            test_labels_df = pd.read_csv(test_file, sep='\t')
            test_indices, test_labels = [], []
            for nod, lab in zip(test_labels_df[nodes_header].values,
                                test_labels_df[label_header].values):
                test_indices.append(nodes_dict[nod])
                test_labels.append(labels_dict[lab])

            test_idx = torch.tensor(test_indices, dtype=torch.long)
            test_y = torch.tensor(test_labels, dtype=torch.long)

            data = Data(edge_index=edge_index, edge_type=edge_type,
                        train_idx=train_idx, train_y=train_y, test_idx=test_idx,
                        test_y=test_y, num_nodes=N)

            if self.hetero:
                data = data.to_heterogeneous(node_type_names=['v'])

            torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.upper()}{self.__class__.__name__}()'


class hide_stdout(object):
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)
