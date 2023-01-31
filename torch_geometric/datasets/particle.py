import glob
import os.path as osp

import numpy as np
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.utils import index_sort, scatter


class TrackingData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'y_index':
            return torch.tensor([value[0].max().item() + 1, self.num_nodes])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        events = glob.glob(osp.join(self.raw_dir, 'event*-hits.csv'))
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events = sorted(events)

    @property
    def raw_file_names(self):
        event_indices = ['000001000']
        file_names = []
        file_names += [f'event{idx}-cells.csv' for idx in event_indices]
        file_names += [f'event{idx}-hits.csv' for idx in event_indices]
        file_names += [f'event{idx}-particles.csv' for idx in event_indices]
        file_names += [f'event{idx}-truth.csv' for idx in event_indices]
        return file_names

    def download(self):
        raise RuntimeError(
            f'Dataset not found. Please download it from {self.url} and move '
            f'all *.csv files to {self.raw_dir}')

    def len(self):
        return len(glob.glob(osp.join(self.raw_dir, 'event*-hits.csv')))

    def get(self, idx):
        import pandas as pd

        idx = self.events[idx]

        # Get hit positions.
        hits_path = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        pos = pd.read_csv(hits_path, usecols=['x', 'y', 'z'], dtype=np.float32)
        pos = torch.from_numpy(pos.values).div_(1000.)

        # Get hit features.
        cells_path = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cell = pd.read_csv(cells_path, usecols=['hit_id', 'value'])
        hit_id = torch.from_numpy(cell['hit_id'].values).to(torch.long).sub_(1)
        value = torch.from_numpy(cell['value'].values).to(torch.float)
        ones = torch.ones(hit_id.size(0))
        num_cells = scatter(ones, hit_id, 0, pos.size(0), 'sum').div_(10.)
        value = scatter(value, hit_id, 0, pos.size(0), 'sum')
        x = torch.stack([num_cells, value], dim=-1)

        # Get ground-truth hit assignments.
        truth_path = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        y = pd.read_csv(truth_path,
                        usecols=['hit_id', 'particle_id', 'weight'])
        hit_id = torch.from_numpy(y['hit_id'].values).to(torch.long).sub_(1)
        particle_id = torch.from_numpy(y['particle_id'].values).to(torch.long)
        particle_id = particle_id.unique(return_inverse=True)[1].sub_(1)
        weight = torch.from_numpy(y['weight'].values).to(torch.float)

        # Sort.
        _, perm = index_sort(particle_id * hit_id.size(0) + hit_id)
        hit_id = hit_id[perm]
        particle_id = particle_id[perm]
        weight = weight[perm]

        # Remove invalid particle ids.
        mask = particle_id >= 0
        hit_id = hit_id[mask]
        particle_id = particle_id[mask]
        weight = weight[mask]

        y_index = torch.stack([particle_id, hit_id], dim=0)

        return TrackingData(x=x, pos=pos, y_index=y_index, y_weight=weight)
