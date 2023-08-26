import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class BrcaTcga(InMemoryDataset):
    r"""The breast cancer (BRCA TCGA) dataset from `cBioPortal
    <https://www.cbioportal.org>`_ and the biological network for node
    connections from `Pathway Commons <https://www.pathwaycommons.org>`_.
    The dataset contains the gene features of each patient in graph_features
    and the overall survival time (in months) of each patient,
    which are the labels.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
        * - 1082
          - 271771
          - 1082
          - 4
    """
    url = 'https://zenodo.org/record/8251328/files/brca_tcga.zip?download=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graph_idx.csv', 'graph_labels.csv', 'edge_index.pt']

    @property
    def processed_file_names(self):
        return 'breast_data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        graph_features = pd.read_csv(
            os.path.join(self.raw_dir, 'brca_tcga', 'graph_idx.csv'),
            index_col=0)
        graph_labels = np.loadtxt(
            os.path.join(self.raw_dir, 'brca_tcga', 'graph_labels.csv'),
            delimiter=',')
        edge_index = torch.load(
            os.path.join(self.raw_dir, 'brca_tcga', 'edge_index.pt'))

        graph_features = graph_features.values
        num_patients = graph_features.shape[0]

        data_list = []
        for i in range(num_patients):
            node_features = graph_features[i]
            target = graph_labels[i]
            data_list.append(
                Data(
                    x=torch.tensor(node_features.reshape(-1, 1)),
                    edge_index=edge_index,
                    y=torch.tensor(target),
                ))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def predefined_split(
            self, train_index, test_index,
            val_index) -> Tuple['BrcaTcga', 'BrcaTcga', 'BrcaTcga']:
        train_dataset = self.index_select(train_index)
        test_dataset = self.index_select(test_index)
        val_dataset = self.index_select(val_index)
        return train_dataset, test_dataset, val_dataset
