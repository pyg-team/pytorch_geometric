import os.path as osp
import glob

import torch
import pandas
import numpy as np
from torch_geometric.data import Data, Dataset


class TrackingData(Data):
    def __inc__(self, key, item):
        if key == 'y':
            return item.max().item() + 1
        else:
            return super(TrackingData, self).__inc__(key, item)


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None):
        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)
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
            'Dataset not found. Please download it from {} and move all '
            '*.csv files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return len(glob.glob(osp.join(self.raw_dir, 'event*-hits.csv')))

    def get(self, idx):
        idx = self.events[idx]

        hits_path = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        pos = pandas.read_csv(hits_path, usecols=['x', 'y', 'z'],
                              dtype=np.float32)
        pos = torch.from_numpy(pos.values)

        truth_path = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        y = pandas.read_csv(truth_path, usecols=['particle_id'],
                            dtype=np.int64)
        y = torch.from_numpy(y.values).squeeze()
        _, y = y.unique(return_inverse=True)

        return TrackingData(pos=pos, y=y)
