from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import *

from graphgym.register import register_loader


def load_dataset_example(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs


register_loader('example', load_dataset_example)
