import gzip
import os

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils.convert import from_networkx


class Email_eu_core(InMemoryDataset):
    r"""This dataset is an e-mail communication network from a large European 
    research institution, taken from the `"Local Higher-order Graph Clustering" <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper. 
    paper <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>.  
    Vertices indicate members of the institution and an edge between a pair 
    of members indicates that they exchanged at least one email. 
    Vertex labels indicate membership to one of the 42 departments.

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

    url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
    url2 = 'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'

from typing import Optional, Callable

    def __init__(self, root: str, transform: Optional[Callable]=None, pre_transform: Optional[Callable]=None):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'email-eu.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # If already downloaded
        if os.path.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
           return
        # Download to `self.raw_dir`.
        for url in [self.url, self.url2]:
            path = download_url(url, self.raw_dir)
            path = os.path.abspath(path)
            with gzip.open(path, 'r') as r:
                with open(os.path.join(self.raw_dir, '.'.join(path.split('.')[:-1])), 'wb') as w:
                    w.write(r.read())
            os.unlink(path)

    def process(self):
        # Read data into huge `Data` list.
        dirname = os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), self.raw_dir)
        filenames = os.listdir(dirname)
        # Order files
        if filenames[0] == 'email-Eu-core-department-labels.txt':
            filenames = filenames[::-1]
        doc1 = open(dirname + '/' + filenames[0], 'rb')

        # Read edge list (same as networkx)
        G = nx.read_edgelist(doc1)

        # Node labels
        try:
            with open(dirname + '/' + filenames[1]) as f:
                for line in f:
                    s = line.strip().split()
                    u = s.pop(0)
                    label = s.pop(0)
                    G.nodes[u]['y'] = int(label)
            #num_unique_node_labels = max(node_labels) + 1
        except IOError:
            print('No node labels')

        # Convert networkx graph to pyg Dataset
        data = from_networkx(G)

        data.num_classes = max(data.y).item() + 1
        data.x = torch.ones(data.num_nodes, 9)
        data.x = torch.cat((data.x, data.y.unsqueeze(1)), dim=1)

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])
