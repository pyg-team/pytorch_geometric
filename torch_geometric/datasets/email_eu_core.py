import os
from typing import Callable, Optional

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from torch_geometric.utils.convert import from_networkx


class EmailEUCore(InMemoryDataset):
    r"""This dataset is an e-mail communication network from a large European 
    research institution, taken from the `"Local Higher-order Graph Clustering" 
    <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper. 
    Vertices indicate members of the institution and an edge between a pair 
    of members indicates that they exchanged at least one email. 
    Vertex labels indicate membership to one of the 42 departments.

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
    """

    url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
    url2 = 'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'

    def __init__(self, root: str, transform: Optional[Callable] = None, 
                                pre_transform: Optional[Callable] = None):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'email-eu.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for url in [self.url, self.url2]:
            path = download_url(url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        # Read data into huge `Data` list.
        filenames = os.listdir(self.raw_dir)
        # Order files
        if filenames[0] == 'email-Eu-core-department-labels.txt':
            filenames = filenames[::-1]
        doc1 = open(self.raw_dir + '/' + filenames[0], 'rb')

        # Read edge list (same as networkx)
        G = nx.read_edgelist(doc1)

        # Node labels
        try:
            with open(self.raw_dir + '/' + filenames[1]) as f:
                for line in f:
                    s = line.strip().split()
                    u = s.pop(0)
                    label = s.pop(0)
                    G.nodes[u]['y'] = int(label)
        except IOError:
            print('No node labels')

        # Convert networkx graph to pyg Dataset
        data = from_networkx(G)
        data.num_classes = max(data.y).item() + 1

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
