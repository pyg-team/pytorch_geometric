import os.path as osp
import glob

import torch
import pandas
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree


class TrackingData(Data):
    def __inc__(self, key, item):
        if key == 'y_index':
            return torch.tensor([item[0].max().item() + 1, self.num_nodes])
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

    def read_hits(self, hits_filename, cells_filename):
        hits = pandas.read_csv(
            hits_filename, usecols=['x', 'y', 'z', 'volume_id', 'layer_id'],
            dtype={
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
            })
        pos = torch.from_numpy(hits[['x', 'y', 'z']].values)
        layer_idx = torch.from_numpy(hits[['volume_id', 'layer_id']].values)

        cells = pandas.read_csv(
            cells_filename, usecols=['hit_id', 'value'], dtype={
                'hit_id': np.int64,
                'value': np.float32,
            })
        hit_id = torch.from_numpy(cells['hit_id'].values).sub_(1)
        value = torch.from_numpy(cells['value'].values)

        num_cells = degree(hit_id, num_nodes=pos.size(0))
        value = scatter_add(value, hit_id, dim_size=pos.size(0))
        x = torch.stack([num_cells, value], dim=-1)

        valid_idx = torch.tensor([[8, 2], [8, 4], [8, 6], [8, 8], [13, 2],
                                  [13, 4], [13, 6], [13, 8], [17, 2], [17, 4]])
        valid_assoc = 20 * valid_idx[:, 0] + valid_idx[:, 1]
        layer_assoc = 20 * layer_idx[:, 0] + layer_idx[:, 1]
        mask = torch.from_numpy(np.isin(layer_assoc, valid_assoc))

        x = x[mask]
        pos = pos[mask]
        layer = layer_assoc[mask].unique(return_inverse=True)[1]

        # mask = torch.zeros(x.size(0), dtype=bool)
        # mask[torch.randperm(x.size(0))[:1000]] = True
        # x = x[mask]
        # pos = pos[mask]
        # layer = layer[mask]

        return mask, x, pos, layer

    def compute_edge_index(self, pos, layer):
        r = (pos[:, 0].pow(2) + pos[:, 1].pow(2)).sqrt()
        phi = torch.atan2(pos[:, 1], pos[:, 0])
        # pos = torch.stack([r, phi, pos[:, 0]], dim=-1)

        phi_slope_max = 0.0006
        z0_max = 100

        edge_indices = []
        for i in range(layer.max().item() + 1):
            mask1 = layer == i
            mask2 = layer == (i + 1)
            nnz1 = mask1.nonzero().flatten()
            nnz2 = mask2.nonzero().flatten()

            dphi = phi[mask2].view(1, -1) - phi[mask1].view(-1, 1)
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi

            adj = dphi
            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)
            return edge_index

            dz = pos[:, 2][mask2].view(1, -1) - pos[:, 2][mask1].view(-1, 1)
            dr = r[mask2].view(1, -1) - r[mask1].view(-1, 1)
            phi_slope = dphi / dr
            z0 = pos[:, 2][mask1].view(-1, 1) - r[mask1].view(-1, 1) * dz / dr
            adj = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)

            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)
            return edge_index

    # dphi = phi2 - phi1
    # dphi[dphi > np.pi] -= 2 * np.pi
    # dphi[dphi < -np.pi] += 2 * np.pi
    # return dphi

    #     print('---------')
    #     print(hits1.values.shape)
    #     print(hits2.values.shape)
    #     print(hit_pairs.values.shape)
    #     print('-------')
    #     # Compute line through the points
    #     dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    #     dz = hit_pairs.z_2 - hit_pairs.z_1
    #     dr = hit_pairs.r_2 - hit_pairs.r_1
    #     phi_slope = dphi / dr
    #     z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    #     # Filter segments according to criteria
    #     good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    #     return hit_pairs[['index_1', 'index_2']][good_seg_mask]
        pass

    def read_y(self, truth_filename, mask, layer):
        truth = pandas.read_csv(
            truth_filename, usecols=['particle_id', 'weight'], dtype={
                'particle_id': np.int64,
                'weight': np.float32
            })

        particle = torch.from_numpy(truth['particle_id'].values)[mask]
        particle = particle.unique(return_inverse=True)[1].sub_(1)
        weight = torch.from_numpy(truth['weight'].values)[mask]

        # edge_indices = []
        # # for i in range(layer.max().item() + 1):
        # for i in range(8, 9):
        #     mask1 = (layer == i) & (particle >= 0)
        #     mask2 = (layer == (i + 1)) & (particle >= 0)
        #     nnz1 = mask1.nonzero().flatten()
        #     nnz2 = mask2.nonzero().flatten()

        #     y = particle[mask1].view(-1, 1) == particle[mask2].view(1, -1)

        #     row, col = y.nonzero().t()
        #     row, col = nnz1[row], nnz2[col]
        #     edge_index = torch.stack([row, col], dim=0)
        #     edge_indices.append(edge_index)

        # edge_index = torch.cat(edge_indices, dim=-1)
        return particle, weight

        # cells = pandas.read_csv(
        #     cells_filename, usecols=['hit_id', 'value'], dtype={
        #         'hit_id': np.int64,
        #         'value': np.float32,
        #     })

        # hit_id = torch.from_numpy(y['hit_id'].values).to(torch.long).sub_(1)
        # particle_id = torch.from_numpy(y['particle_id'].values).to(torch.long)
        # particle_id = particle_id.unique(return_inverse=True)[1].sub_(1)
        # weight = torch.from_numpy(y['weight'].values).to(torch.float)

    def read_event(self, idx):
        idx = self.events[idx]

        hits_filename = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        cells_filename = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        mask, x, pos, layer = self.read_hits(hits_filename, cells_filename)

        print(mask.size(), x.size(), pos.size(), layer.size())
        print(mask[:2])
        print(x[:2])
        print(pos[:2])
        print(layer[:2])

        # edge_index = self.compute_edge_index(pos, layer)
        truth_filename = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        particle, weight = self.read_y(truth_filename, mask, layer)

        return Data(x=x, pos=pos, layer=layer, particle=particle,
                    weight=weight)

        # print(layer_assoc)
        # print(layer[:5])

        # print('---------')
        # count = 0
        # for i in range(valid_idx.size(0)):
        #     mask1 = layer[:, 0] == valid_idx[i, 0]
        #     mask2 = layer[:, 1] == valid_idx[i, 1]
        #     mask = mask1 & mask2
        #     print(mask.sum())
        #     count += mask.sum().item()
        # print('---------')

        # print(bla.shape, bla.sum())

        # # print(layer[:, 0].min(), layer[:, 0].max())
        # # print(layer[:, 1].min(), layer[:, 1].max())
        # print(bla)

        # print(layer)

        raise NotImplementedError

        pos = torch.from_numpy(pos.values).div_(1000.)

        # Get hit features.
        cells_path = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cell = pandas.read_csv(cells_path, usecols=['hit_id', 'value'])
        hit_id = torch.from_numpy(cell['hit_id'].values).to(torch.long).sub_(1)
        value = torch.from_numpy(cell['value'].values).to(torch.float)
        ones = torch.ones(hit_id.size(0))
        num_cells = scatter_add(ones, hit_id, dim_size=pos.size(0)).div_(10.)
        value = scatter_add(value, hit_id, dim_size=pos.size(0))
        x = torch.stack([num_cells, value], dim=-1)

        # Get ground-truth hit assignments.
        truth_path = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        y = pandas.read_csv(truth_path,
                            usecols=['hit_id', 'particle_id', 'weight'])
        hit_id = torch.from_numpy(y['hit_id'].values).to(torch.long).sub_(1)
        particle_id = torch.from_numpy(y['particle_id'].values).to(torch.long)
        particle_id = particle_id.unique(return_inverse=True)[1].sub_(1)
        weight = torch.from_numpy(y['weight'].values).to(torch.float)

        # Sort.
        perm = (particle_id * hit_id.size(0) + hit_id).argsort()
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

    def get(self, idx):
        return self.read_event(idx)
