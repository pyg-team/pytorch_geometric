import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class PascalPF(InMemoryDataset):
    r"""The Pascal-PF dataset from the `"Proposal Flow"
    <https://arxiv.org/abs/1511.05065>`_ paper, containing 4 to 16 keypoints
    per example over 20 categories.

    Args:
        root (str): Root directory where the dataset should be saved.
        category (str): The category of the images (one of
            :obj:`"Aeroplane"`, :obj:`"Bicycle"`, :obj:`"Bird"`,
            :obj:`"Boat"`, :obj:`"Bottle"`, :obj:`"Bus"`, :obj:`"Car"`,
            :obj:`"Cat"`, :obj:`"Chair"`, :obj:`"Diningtable"`, :obj:`"Dog"`,
            :obj:`"Horse"`, :obj:`"Motorbike"`, :obj:`"Person"`,
            :obj:`"Pottedplant"`, :obj:`"Sheep"`, :obj:`"Sofa"`,
            :obj:`"Train"`, :obj:`"TVMonitor"`)
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
    """
    url = ('https://www.di.ens.fr/willow/research/proposalflow/dataset/'
           'PF-dataset-PASCAL.zip')

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(
        self,
        root: str,
        category: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.category = category.lower()
        assert self.category in self.categories
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])
        self.pairs = fs.torch_load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> List[str]:
        return ['Annotations', 'parsePascalVOC.mat']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.category}.pt', f'{self.category}_pairs.pt']

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        fs.rm(self.raw_dir)
        os.rename(osp.join(self.root, 'PF-dataset-PASCAL'), self.raw_dir)

    def process(self) -> None:
        from scipy.io import loadmat

        path = osp.join(self.raw_dir, 'Annotations', self.category, '*.mat')
        filenames = glob.glob(path)

        names = []
        data_list = []
        for filename in filenames:
            name = osp.basename(filename).split('.')[0]

            pos = torch.from_numpy(loadmat(filename)['kps']).to(torch.float)
            mask = ~torch.isnan(pos[:, 0])
            pos = pos[mask]

            # Normalize points to unit sphere.
            pos = pos - pos.mean(dim=0, keepdim=True)
            pos = pos / pos.norm(dim=1).max()

            y = mask.nonzero(as_tuple=False).flatten()

            data = Data(pos=pos, y=y, name=name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            names.append(name)
            data_list.append(data)

        pairs = loadmat(osp.join(self.raw_dir, 'parsePascalVOC.mat'))
        pairs = pairs['PascalVOC']['pair'][0, 0][
            0, self.categories.index(self.category)]

        pairs = [(names.index(x[0][0]), names.index(x[1][0])) for x in pairs]

        self.save(data_list, self.processed_paths[0])
        torch.save(pairs, self.processed_paths[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')
