"""The module contains definition of logic for processing InLUT3D dataset.

The InLUT3D dataset was released under the Creative Commons Attribution
Non Commercial 2.5 Generic Licese and is publicly available under the
following link: https://doi.org/10.5281/zenodo.8131487
"""

__author__ = "Jakub Walczak"
__email__ = "jakub.walczak@p.lodz.pl"

import os
import os.path as osp
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import Data, Dataset, download_url, extract_tar_gz
from torch_geometric.io import fs

_PTS_SKIP_ROW_NBR: int = 1
_XYZ_COLS_IDS: tuple = (0, 1, 2)
_RGB_COLS_IDS: tuple = (3, 4, 5)
_CATEGORY_COLS_ID: tuple = (6, )
_INSTANCE_COLS_ID: tuple = (7, )


def read_inlut3d_pts(path: str, ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Read PTS file containing point cloud data of InLUT3D dataset."""
    xyz = np.loadtxt(
        path,
        usecols=_XYZ_COLS_IDS,
        skiprows=_PTS_SKIP_ROW_NBR,
        dtype=np.float32,
    )
    rgb = np.loadtxt(path, usecols=_RGB_COLS_IDS, skiprows=_PTS_SKIP_ROW_NBR,
                     dtype=np.int32)
    categories = np.loadtxt(
        path,
        usecols=_CATEGORY_COLS_ID,
        skiprows=_PTS_SKIP_ROW_NBR,
        dtype=np.int32,
    )
    instances = np.loadtxt(
        path,
        usecols=_INSTANCE_COLS_ID,
        skiprows=_PTS_SKIP_ROW_NBR,
        dtype=np.int32,
    )
    return (torch.from_numpy(xyz), torch.from_numpy(rgb),
            torch.from_numpy(categories), torch.from_numpy(instances))


def _get_setup_name_from_path(path: str) -> str:
    return osp.splitext(osp.basename(path))[0]


class InLUT3D(Dataset):
    r"""The Indoor Lodz University of Technology Point Cloud Dataset (InLUT3D)
    containing 321 point clouds of indoor setups across various buildings
    of W7 faculty of Lodz University of Technology. Points are annotated with
    18 distinct categories. Each point (a node) is described by its
    3D position, unnormalised color in RGB space, a category, and an instance
    codes.

    Args:
        root (str): Root directory where the dataset should be saved.
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
    """

    URL: str = "https://zenodo.org/records/8131487/files/inlut3d.tar.gz"
    SAMPLES_NBR: int = 321

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self._split_processed_paths = list(
            filter(
                lambda file:
                (_get_setup_name_from_path(file) in self.train_setups if train
                 else _get_setup_name_from_path(file) in self.train_setups),
                self.processed_paths,
            ))

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join(f"setup_{i}", f"setup_{i}.pts")
            for i in range(self.SAMPLES_NBR)
        ] + ["train.txt", "test.txt", "labels.json"]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"setup_{i}.pt" for i in range(self.SAMPLES_NBR)]

    @property
    def train_setups(self) -> List[str]:
        with open(osp.join(self.raw_dir, "train.txt")) as f:
            return [
                _get_setup_name_from_path(file)
                for file in f.read().splitlines()
            ]

    @property
    def test_files(self) -> List[str]:
        with open(osp.join(self.raw_dir, "train.txt")) as f:
            return [
                _get_setup_name_from_path(file)
                for file in f.read().splitlines()
            ]

    @property
    def labels(self) -> dict:
        import json

        with open(osp.join(self.raw_dir, "labels.json")) as f:
            return json.load(f)

    def download(self) -> None:
        path = download_url(self.URL, self.root)
        extract_tar_gz(path, self.raw_dir)
        os.unlink(path)

    def len(self) -> int:
        return len(self._split_processed_paths)

    def get(self, i: int) -> Data:
        return fs.torch_load(self._split_processed_paths[i])

    def process(self) -> None:
        for filename in tqdm(self.raw_paths):
            setup_name, _ = osp.splitext(osp.basename(filename))
            xyz, rgb, categories, instances = read_inlut3d_pts(filename)
            data = Data(pos=xyz, x=rgb, cat=categories, instances=instances)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            fs.torch_save(data, osp.join(self.processed_dir,
                                         f"{setup_name}.pt"))
