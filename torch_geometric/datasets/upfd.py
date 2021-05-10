import os.path as osp
import numpy as np
import scipy.sparse as sp
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array


class UPFD(InMemoryDataset):
    r"""The tree-structured fake news propagation graph classification
    dataset from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake&real news propagation
    graphs extracted from Twitter.
    For a single graph, the root node represents the source news,
    leaf nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user
    retweeted the root news directly.
    Two user nodes have an edge if and only if one user retweeted
    the root news from the other user.
    Four different node features are encoded using different encoders.
    Refer to `GNN-FakeNews <https://github.com/safe-graph/GNN-FakeNews>`_
    repo for more details.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the graph set, (:obj:`"politifact"`,
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

    pol_id = '1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-'
    gos_id = '1VskhAQ92PrT4sWEKQ2v2-AJhEcpp4A81'

    def __init__(self, root, name, feature,
                 transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        self.feature = feature
        super(UPFD, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, self.train_idx, self.val_idx, self.test_idx\
            = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names] + \
               ['A.txt', f'new_{self.feature}_feature.npz']

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def download(self):
        shutil.rmtree(self.raw_dir)
        if self.name == 'politifact':
            gdd.download_file_from_google_drive(self.pol_id, self.raw_dir +
                                                f'{self.name}.zip', unzip=True)
        else:
            gdd.download_file_from_google_drive(self.gos_id, self.raw_dir +
                                                f'{self.name}.zip', unzip=True)

    def process(self):

        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # The fixed data split for benchmarking evaluation
        # train-val-test split is 20%-10%-70%
        self.train_idx = torch.from_numpy(np.load(
            self.raw_dir + 'train_idx.npy')).to(torch.long)
        self.val_idx = torch.from_numpy(np.load(
            self.raw_dir + 'val_idx.npy')).to(torch.long)
        self.test_idx = torch.from_numpy(np.load(
            self.raw_dir + 'test_idx.npy')).to(torch.long)

        torch.save((self.data, self.slices, self.train_idx,
                    self.val_idx, self.test_idx), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


def split(data, batch):
    """
    Util code to create graph batches
    """

    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def read_graph_data(folder, feature):
    """
    Util code to create PyG data instance from raw graph data
    """

    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
    edge_index = read_txt_array(osp.join(folder, 'A.txt'),
                                sep=',', dtype=torch.long).t()
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')

    edge_attr = None
    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr,
                                     num_nodes, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, node_graph_id)

    return data, slices
