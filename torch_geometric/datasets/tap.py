from typing import Callable, Optional, Tuple, Union

import torch

from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
)


def read_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f)


def parse_npz(f):
    crash_time = f['crash_time']
    x = torch.from_numpy(f['x']).to(torch.float)
    coords = torch.from_numpy(f['coordinates']).to(torch.float)
    edge_attr = torch.from_numpy(f['edge_attr']).to(torch.float)
    cnt_labels = torch.from_numpy(f['cnt_labels']).to(torch.long)
    occur_labels = torch.from_numpy(f['occur_labels']).to(torch.long)
    edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
    edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
    severity_labels = torch.from_numpy(f['severity_8labels']).to(torch.long)
    edge_index = torch.from_numpy(f['edge_index']).to(
        torch.long).t().contiguous()
    return Data(x=x, y=occur_labels, severity_labels=severity_labels,
                edge_index=edge_index, edge_attr=edge_attr,
                edge_attr_dir=edge_attr_dir, edge_attr_ang=edge_attr_ang,
                coords=coords, cnt_labels=cnt_labels, crash_time=crash_time)


class TRAVELDataset(InMemoryDataset):
    r"""The Traffic Accident Prediction (TAP) dataset introduced in the
    `"TAP: A Comprehensive Data Repository for Traffic Accident Prediction in Road Networks"
    <https://arxiv.org/pdf/2304.08640>`_ paper.
    Nodes represent intersections and edges are roads.
    Node and edge features represent embeddings of geospatial features.
    The task is to predict the occurrence and severity of accidents on roadways.
    For further information, please refer to the TAP repository.
    `TAP
    <https://github.com/baixianghuang/travel>`_

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://github.com/baixianghuang/travel/raw/main/TAP-city/{}.npz'

    # url = 'https://github.com/baixianghuang/travel/raw/main/TAP-state/{}.npz'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'
