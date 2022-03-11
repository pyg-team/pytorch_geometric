import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array


class UPFD(InMemoryDataset):
    r"""The tree-structured fake news propagation graph classification dataset
    from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake & real news propagation graphs
    extracted from Twitter.
    For a single graph, the root node represents the source news, and leaf
    nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user retweeted
    the root news directly.
    Two user nodes have an edge if and only if one user retweeted the root news
    from the other user.
    Four different node features are encoded using different encoders.
    Please refer to `GNN-FakeNews
    <https://github.com/safe-graph/GNN-FakeNews>`_ repo for more details.

    .. note::

        For an example of using UPFD, see `examples/upfd.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        upfd.py>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the graph set (:obj:`"politifact"`,
            :obj:`"gossipcop"`).
        feature (string): The node feature type (:obj:`"profile"`,
            :obj:`"spacy"`, :obj:`"bert"`, :obj:`"content"`).
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
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
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    ids = {
        'politifact': '1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-',
        'gossipcop': '1VskhAQ92PrT4sWEKQ2v2-AJhEcpp4A81',
    }

    def __init__(self, root, name, feature, split="train", transform=None,
                 pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        self.feature = feature
        super().__init__(root, transform, pre_transform, pre_filter)

        assert split in ['train', 'val', 'test']
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed', self.feature)

    @property
    def raw_file_names(self):
        return [
            'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
            'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz'
        ]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url.format(self.ids[self.name]), self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        x = sp.load_npz(
            osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
        x = torch.from_numpy(x.todense()).to(torch.float)

        edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        node_slice = torch.cumsum(batch.bincount(), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])
        edge_slice = torch.cumsum(batch[edge_index[0]].bincount(), 0)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice
        }

        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(x=x, edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            torch.save(self.collate(data_list), path)

    def __repr__(self):
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')
