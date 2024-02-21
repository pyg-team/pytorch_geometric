import json
import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class AirfRANS(InMemoryDataset):
    r"""The AirfRANS dataset from the `"AirfRANS: High Fidelity Computational
    Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes
    Solutions" <https://arxiv.org/abs/2212.07564>`_ paper, consisting of 1,000
    simulations of steady-state aerodynamics over 2D airfoils in a subsonic
    flight regime.
    The different tasks (:obj:`"full"`, :obj:`"scarce"`, :obj:`"reynolds"`,
    :obj:`"aoa"`) define the utilized training and test splits.

    Each simulation is given as a point cloud defined as the nodes of the
    simulation mesh. Each point of a point cloud is described via 5
    features: the inlet velocity (two components in meter per second), the
    distance to the airfoil (one component in meter), and the normals (two
    components in meter, set to :obj:`0` if the point is not on the airfoil).
    Each point is given a target of 4 components for the underyling regression
    task: the velocity (two components in meter per second), the pressure
    divided by the specific mass (one component in meter squared per second
    squared), the turbulent kinematic viscosity (one component in meter squared
    per second).
    Finaly, a boolean is attached to each point to inform if this point lies on
    the airfoil or not.

    A library for manipulating simulations of the dataset is available `here
    <https://airfrans.readthedocs.io/en/latest/index.html>`_.

    The dataset is released under the `ODbL v1.0 License
    <https://opendatacommons.org/licenses/odbl/1-0/>`_.

    .. note::

        Data objects contain no edge indices to be agnostic to the simulation
        mesh. You are free to build a graph via the
        :obj:`torch_geometric.transforms.RadiusGraph` transform.

    Args:
        root (str): Root directory where the dataset should be saved.
        task (str): The task to study (:obj:`"full"`, :obj:`"scarce"`,
            :obj:`"reynolds"`, :obj:`"aoa"`) that defines the utilized training
            and test splits.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 1,000
          - ~180,000
          - 0
          - 5
          - 4
    """
    url = 'https://data.isir.upmc.fr/extrality/pytorch_geometric/AirfRANS.zip'
    tasks = ['full', 'scarce', 'reynolds', 'aoa']

    def __init__(
        self,
        root: str,
        task: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if task not in self.tasks:
            raise ValueError(f"Expected 'task' to be in {self.tasks} "
                             f"got '{task}'")

        self.task = 'full' if task == 'scarce' and not train else task
        self.split = 'train' if train else 'test'

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['AirfRANS.pt', 'manifest.json']

    @property
    def processed_file_names(self) -> str:
        return f'{self.task}_{self.split}.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[1], 'r') as f:
            manifest = json.load(f)
        total = manifest['full_train'] + manifest['full_test']
        partial = set(manifest[f'{self.task}_{self.split}'])

        data_list = []
        raw_data = torch.load(self.raw_paths[0])
        for k, s in enumerate(total):
            if s in partial:
                data = Data(**raw_data[k])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'task={self.task}, split={self.split})')
