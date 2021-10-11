from typing import Optional, Callable
import os.path as osp
import torch
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data


class Airports(InMemoryDataset):
    r"""The Airports dataset from the `"struc2vec: Learning Node
    Representations from Structural Identity"
    <https://arxiv.org/abs/1704.03165>`_ paper, where nodes denote airports
    and labels correspond to activity levels.
    The feature generating method (one-hot encoding) is from the
    `"GraLSP: Graph Neural Networks with Local Structural Patterns"
    ` <https://arxiv.org/abs/1911.07675>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"usa"`,
            :obj:`"brazil"`, :obj:`"europe"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    base_edge_list_url = "https://github.com/leoribeiro/struc2vec"\
        "/raw/master/graph/{}-airports.edgelist"
    base_label_url = "https://github.com/leoribeiro/struc2vec"\
        "/raw/master/graph/labels-{}-airports.txt"

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']
        self.edge_list_url = self.base_edge_list_url.format(self.name)
        self.edge_list_file = "{}-airports.edgelist".format(self.name)
        self.label_url = self.base_label_url.format(self.name)
        self.label_file = "labels-{}-airports.txt".format(self.name)
        super(Airports, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [self.label_file, self.edge_list_file]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.label_url, self.raw_dir)
        download_url(self.edge_list_url, self.raw_dir)

    def process(self):
        index_map = dict()
        with open(osp.join(self.raw_dir, self.label_file), 'r') as f:
            data = f.read().split('\n')[1:-1]
            i, n = 0, len(data)
            x, y = list(), list()
            for pair in data:
                a, b = pair.split(' ')
                # generate unique one-hot feature for each node
                feat_vector = [0.0] * i + [1.0] + [0.0] * (n - i - 1)
                index_map[int(a)] = i
                x.append(feat_vector)
                y.append(int(b))
                i += 1
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)

        with open(osp.join(self.raw_dir, self.edge_list_file), 'r') as f:
            data = list()
            for pair in f.read().split('\n'):
                split = pair.split(' ')
                if len(split) == 2:
                    data.append([index_map[int(split[0])],
                                index_map[int(split[1])]])
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y, index_map=[
                    [k, index_map[k]] for k in index_map.keys()])
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}Airports()'.format(self.name.capitalize())
