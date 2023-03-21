import os.path as osp
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset

NAME = "GRAPHSAT"


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, name=None, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [NAME + ".pkl"]

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        # data_list = pickle.load(open(osp.join(self.root, "raw/" + NAME + ".pkl"), "rb"))
        cur_dir = osp.dirname(osp.realpath(__file__))
        f = osp.join(cur_dir, f'./Data/{self.name}', "raw/" + NAME + ".pkl")
        data_list = pickle.load(open(f, "rb"))

        for data in data_list:
            n_edge = data.edge_index.size(1)
            data.edge_attr = torch.ones((n_edge, 1))  # added a fake edge_attr for EXP/CEXP dataset
            data.x = data.x.type(torch.float32)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_tasks(self):
        return 1  # it is always binary classification for the datasets we consider

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    def separate_data(self, seed, fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test
        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}


def viz_graph(g, node_size=5, edge_width=1, node_color='b', show=False):
    pos = nx.planar_layout(g)
    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width=edge_width)
    if show: plt.show()
