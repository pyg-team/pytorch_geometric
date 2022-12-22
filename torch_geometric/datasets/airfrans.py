import json
import os
import warnings
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
    Solutions" <https://arxiv.org/abs/2212.07564>`_ paper, consisting of 1000
    simulations of steady-state aerodynamics over 2D airfoils in a subsonic
    flight regime. Different tasks are proposed, they define the split between
    training and test set. See the paper for more information on their definitions.

    Each simulation is given as a point cloud defined as the nodes of the
    simulation mesh. Each point of a point cloud is given 5
    features: the inlet velocity (two components in meters per second),
    the distance to the airfoil (in meter) and the normals (two
    components, set to 0 if the point is not on the airfoil); it is also
    given its Cartesian position (in meter). As it is a regression task,
    each point is given a target of 4 components: the velocity (two components
    in meter per second), the pressure (divided by the specific mass, in meter
    squared per second squared), the turbulent kinematic viscosity (in meter
    squared per second). Finaly, a boolean is attached to each point to
    inform if this point lies on the airfoil or not.

    .. note::

        Data objects contain no edge indices as we would like to be agnostic
        to the simulation mesh. You are free to build a graph via the
        :obj:`torch_geometric.transforms.RadiusGraph` transform, with its cut-off
        being a hyperparameter.

    Args:
        root (string): Root directory where the dataset should be saved.
        task (string): Task to study. It defines the split of the training and test set.
        train (bool, optional): Determines whether the train or test split of the associated task
            gets loaded.
            (default: :obj:`True`)
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

    Stats:
        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #simulations
              - #nodes
              - #features
              - #targets
            * - 1000
              - ~180000
              - 5 (+2)
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
    ):

        assert task in self.tasks

        if task == 'scarce' and not train:
            self.task = 'full'
            warnings.warn(
                'Task has been replaced by "full" as it is the same test set for both tasks.',
                UserWarning)
        else:
            self.task = task
        self.split = train * 'train' + (1 - train) * 'test'

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['AirfRANS.pt', 'manifest.json']

    @property
    def processed_file_names(self) -> List[str]:
        return [self.task + '_' + self.split + '.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[1], 'r') as f:
            manifest = json.load(f)
        total = manifest['full_train'] + manifest['full_test']
        partial = manifest[self.task + '_' + self.split]

        raw_data = torch.load(self.raw_paths[0])
        data_list = []
        for k, s in enumerate(total):
            if bool(set([s]) & set(partial)):
                data = Data(**raw_data[k])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'task = {self.task}, '
                f'split = {self.split})')
