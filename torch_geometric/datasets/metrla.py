from typing import Optional, Callable

from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.io import read_npz
import pandas as pd
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch_geometric.io.metrla import MetrLaIo
from typing import Optional
from torch_geometric.utils.sparse import dense_to_sparse


class MetrLa(Dataset):
    r"""The Los Angeles Metropolitan (MetrLA) traffic dataset from the the highways introduced in the
    `"Big Data and its Technical Challenges"
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.681.7248&rep=rep1&type=pdf>`_ paper, and brought to
    attention by papers that promote application of graph neural networks on traffic data, (i.e.
    `"DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING"
    <https://arxiv.org/pdf/1707.01926.pdf>`_)

    Each node represents a sensor in a network, which can detect cars passing by it. The measurements are aggregated
    in 5-minutes intervals. This is a time-series forecasting task on graph data, so the features of each node have
    an additional, temporal dimension.

    The initial adjacency matrix is constructed by applying a thresholded Gaussian kernel over the real-world distances
    between the sensors:

    :math: W_{ij} = exp(-\frac{dist(v_{i}, v_{j})^2}{\sigma^{2}}), if dist(v_{i}, v_{j}) >= K, otherwise 0, where K
    is a pre-fixed threshold, and dist(, ) is the real-world road distance between two sensors, and :math: \sigma^{2}
    is the variance of the distances between sensors. A zero-value in the adjacency matrix does not indicate that there
    is no road connection between two sensors (because there almost always is, since road networks are connected graphs.
    Instead, it means that the distance between them is larger than the threshold value.

    The traffic prediciton task is a time series regression, which aims to predict the traffic volumen values in each
    node for the next *n_next* time steps, given the volumnes in the *n_prev* previous time steps. Both n_next and
    n_prev are pre-set values.  

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
    gdrive_ids = {'distances.csv': '19Td6JafGnF8CD2H64jWCBV7k5GyFix2H',
                  'locations.csv': '1FnioVF2jZuOl_St1ssLnvQgSHuEuDzvL',
                  'sensors.txt': '1235bXBxe4X73dJk3zwRzaIQ-8A7K_OEV',
                  'sensor-readings.h5': '1VgLjs4XB5O-IpcyZ5GAQG9sGuyxKe3Lc'}

    def __init__(self,
                 root: str,
                 n_previous_steps: int,
                 n_future_steps: int,
                 add_time_of_day: bool = False,
                 add_day_of_week: bool = False,
                 normalized_k: float = .1,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None) -> None:

        # TODO
        # This is sort of a circular dependency, so hard coding it fixes it.
        self.n_readings = 34272

        self.io = MetrLaIo(n_readings=self.n_readings,
                           n_previous_steps=n_previous_steps,
                           n_future_steps=n_future_steps,
                           add_time_of_day=add_time_of_day,
                           add_day_of_week=add_day_of_week,
                           normalized_k=normalized_k)

        self.min_t = self.io.min_t
        self.max_t = self.io.max_t
        self.dataset_len = self.io.dataset_len

        super().__init__(root, transform, pre_transform)

        self.normalized_k = normalized_k

    @property
    def raw_file_names(self) -> str:
        """
        Return the list of file names in the raw folder.

        :return: A list of file names.
        """
        return list(self.gdrive_ids.keys())

    @property
    def processed_file_names(self) -> str:
        """
        
        :return: 
        """
        return [f'data_{i}.pt' for i in range(self.dataset_len)]

    def download(self) -> None:
        for file_name, gdrive_id in self.gdrive_ids.items():
            gdd.download_file_from_google_drive(file_id=gdrive_id,
                                                dest_path=Path(self.raw_dir) / file_name)

    def process(self) -> None:
        self.io.load_metrla_data(data_path=self.raw_paths[3])

        self.io.generate_adjacency_matrix(distances_path=self.raw_paths[0],
                                          sensor_ids_path=self.raw_paths[2])

        x, y = self.io.data

        # Generate the adjacency matrix, which is constant throughout the measurements.
        adjacency_matrix = torch.tensor(data=self.io.adjacency_matrix, dtype=torch.float32)

        edge_index, edge_attributes = dense_to_sparse(adjacency_matrix)

        for idx in range(self.dataset_len):
            # Select the "slice" among the first dimension - This pair represents a single training example.
            # To be wrapped in a "Data" object.

            # x_index (n_previous_steps, n_nodes, n_features)
            x_index = torch.tensor(data=x[idx, ...], dtype=torch.float32)

            # y_index (n_next_steps, n_nodes, n_features)
            y_index = torch.tensor(data=x[idx, ...], dtype=torch.float32)

            data = Data(x=x_index,
                        edge_index=edge_index,
                        edge_attributes=edge_attributes,
                        y=y_index)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            file_name = self.processed_file_names[idx]
            torch.save(data, Path(self.processed_dir) / file_name)

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        file_name = self.processed_file_names[idx]
        data = torch.load(Path(self.processed_dir) / file_name)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'
