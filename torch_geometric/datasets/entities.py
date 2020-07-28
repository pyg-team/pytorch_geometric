import os
import os.path as osp
from collections import Counter

import gzip
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)


class Entities(InMemoryDataset):
    r"""The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from
    the `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"AIFB"`,
            :obj:`"MUTAG"`, :obj:`"BGS"`, :obj:`"AM"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{}.tgz'

    def __init__(self, root, name, transform=None, pre_transform=None):
        assert name in ['AIFB', 'AM', 'MUTAG', 'BGS']
        self.name = name.lower()
        super(Entities, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self):
        return self.data.train_y.max().item() + 1

    @property
    def raw_file_names(self):
        return [
            '{}_stripped.nt.gz'.format(self.name),
            'completeDataset.tsv',
            'trainingSet.tsv',
            'testSet.tsv',
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url.format(self.name), self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def triples(self, graph, relation=None):
        for s, p, o in graph.triples((None, relation, None)):
            yield s, p, o

    def process(self):
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        g = rdf.Graph()
        with gzip.open(graph_file, 'rb') as f:
            g.parse(file=f, format='nt')

        freq_ = Counter(g.predicates())

        def freq(rel):
            return freq_[rel] if rel in freq_ else 0

        relations = sorted(set(g.predicates()), key=lambda rel: -freq(rel))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        relations_dict = {rel: i for i, rel in enumerate(list(relations))}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edge_list = []
        for s, p, o in g.triples((None, None, None)):
            src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
            edge_list.append([src, dst, 2 * rel])
            edge_list.append([dst, src, 2 * rel + 1])

        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2]

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

        data = Data(edge_index=edge_index)
        data.edge_type = edge_type
        data.train_idx = train_idx
        data.train_y = train_y
        data.test_idx = test_idx
        data.test_y = test_y
        data.num_nodes = edge_index.max().item() + 1

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
