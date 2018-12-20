import os
import os.path as osp
import rdflib as rdf
from collections import Counter
from torch_sparse import coalesce

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_gz


class Entities(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        assert name in ['AIFB', 'AM', 'MUTAG', 'BGS']
        self.name = name
        super(Entities, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'mutag_stripped.nt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = 'https://www.dropbox.com/s/qy8j3p8eacvm4ir/'\
              'mutag_stripped.nt.gz?dl=1'
        path = download_url(url, self.root)
        extract_gz(path, self.raw_dir, 'mutag_stripped.nt')
        os.unlink(path)

    def freq(self, freq, relation):
        if relation not in freq:
            return 0
        return freq[relation]

    def triples(self, graph, relation=None):
        for s, p, o in graph.triples((None, relation, None)):
            yield s, p, o

    def process(self):
        graph = rdf.Graph()
        with open(osp.join(self.raw_dir, 'mutag_stripped.nt'), 'rb') as f:
            graph.parse(file=f, format='nt')
        freq = Counter(graph.predicates())

        relations = list(set(graph.predicates()))
        relations.sort(key=lambda rel: -self.freq(freq, rel))

        subjects = set(graph.subjects())
        objects = set(graph.objects())

        nodes = list(subjects.union(objects))
        # adj_shape = (len(nodes), len(nodes))

        # relations_dict = {rel: i for i, rel in enumerate(list(relations))}
        nodes_dict = {node: i for i, node in enumerate(nodes)}
        # adjacencies = []

        size = 0
        all_edges = []
        for i, rel in enumerate(relations):
            print('relation {}: {}, frequency {}'.format(i, rel,
                                                         self.freq(freq, rel)))
            edges = torch.empty((self.freq(freq, rel), 2), dtype=torch.long)

            for j, (s, p, o) in enumerate(self.triples(graph, rel)):
                if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                    print(s, o, nodes_dict[s], nodes_dict[o])

                edges[j, 0] = nodes_dict[s]
                edges[j, 1] = nodes_dict[o]
                size += 1
            all_edges.append(edges)

        edges = torch.cat(all_edges, dim=0)

        edges, _ = coalesce(edges.t().contiguous(), None,
                            len(nodes), len(nodes))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
