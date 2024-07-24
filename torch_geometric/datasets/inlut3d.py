"""The module contains definition of logic for processing InLUT3D dataset.

The InLUT3D dataset was released under the Creative Commons Attribution
Non Commercial 2.5 Generic Licese and is publicly available under the
following link: https://zenodo.org/doi/10.5281/zenodo.8131486
"""

__author__ = "Jakub Walczak"
__email__ = "jakub.walczak@p.lodz.pl"

import os
import os.path as osp
from typing import Callable, List, Literal, Optional, Tuple, Type

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
    return (
        torch.from_numpy(xyz),
        torch.from_numpy(rgb),
        torch.from_numpy(categories.astype(int)),
        torch.from_numpy(instances.astype(int)),
    )


class InLUT3D(Dataset):
    r"""The Indoor Lodz University of Technology Point Cloud Dataset (InLUT3D)
    of indoor setups across various buildings of W7 faculty of Lodz University
    of Technology. Points are annotated with 18 distinct categories.
    Each point (a node) is described by its 3D position, unnormalised color in
    RGB space, a category, and instance codes.

    Args:
        task (str): The task for which the dataset should be used. It can be
            one of the following: "classification", "semantic_segmentation",
            "instance_segmentation".
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
        remove_unlabeled (bool, optional): Whether to remove points with
            unlabeled categories. (default: :obj:`True`)
    """

    URL: str = "https://zenodo.org/records/12804009/files/inlut3d.tar.gz"
    SETUPS_NBR: int = 321
    FIRST_TEST_ID: int = 301

    def __new__(
        cls: Type["InLUT3D"],
        task: Literal["classification", "semantic_segmentation",
                      "instance_segmentation"],
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        remove_unlabeled: bool = True,
    ) -> "InLUT3D":
        if task == "classification":
            return object.__new__(ClassificationInLUT3D)
        elif task == "semantic_segmentation":
            return object.__new__(SemanticSegmentationInLUT3D)
        elif task == "instance_segmentation":
            return object.__new__(InstanceSegmentationInLUT3D)
        else:
            raise ValueError(f"InLU3D cannot be used for: {task}")

    def __init__(
        self,
        task: Literal["classification", "semantic_segmentation",
                      "instance_segmentation"],
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        remove_unlabeled: bool = True,
    ) -> None:
        self.remove_unlabeled = remove_unlabeled
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        if train:
            self.split_processed_paths = self.collect_train_samples()
        else:
            self.split_processed_paths = self.collect_test_samples()

    def collect_train_samples(self) -> Tuple[str]:
        raise NotImplementedError

    def collect_test_samples(self) -> Tuple[str]:
        raise NotImplementedError

    @property
    def train_setups(self) -> List[str]:
        return [f"setup_{i}" for i in range(self.FIRST_TEST_ID)]

    @property
    def test_setups(self) -> List[str]:
        return [
            f"setup_{i}" for i in range(self.FIRST_TEST_ID, self.SETUPS_NBR)
        ]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join(f"setup_{i}", f"setup_{i}.pts")
            for i in range(self.SETUPS_NBR)
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
        return len(self.split_processed_paths)

    def get(self, i: int) -> Data:
        return fs.torch_load(self.split_processed_paths[i])

    def process(self) -> None:
        raise NotImplementedError

    def iter_over_setups(self):  # type: ignore[no-untyped-def]
        for filename in tqdm(self.raw_paths):
            setup_name, _ = osp.splitext(osp.basename(filename))
            xyz, rgb, categories, instances = read_inlut3d_pts(filename)
            if self.remove_unlabeled:
                mask = categories != -1
                xyz, rgb, categories, instances = (
                    xyz[mask],
                    rgb[mask],
                    categories[mask],
                    instances[mask],
                )
                yield setup_name, (xyz, rgb, categories, instances)


class ClassificationInLUT3D(InLUT3D):
    SAMPLES_NBR: int = 3608
    FIRST_TEST_ID: int = 2801

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_classification")

    @property
    def processed_file_names(self) -> List[str]:
        return [f"sample_{i}.pt" for i in range(self.SAMPLES_NBR)]

    def collect_train_samples(self) -> Tuple[str]:
        return tuple([
            osp.join(self.processed_dir, f"sample_{i}.pt")
            for i in range(self.FIRST_TEST_ID)
        ])

    def collect_test_samples(self) -> Tuple[str]:
        return tuple([
            osp.join(self.processed_dir, f"sample_{i}.pt")
            for i in range(self.FIRST_TEST_ID, self.SAMPLES_NBR)
        ])

    def _samples_ids_gen(self):  # type: ignore[no-untyped-def]
        id_ = 0
        while True:
            yield id_
            id_ += 1

    def process(self) -> None:
        id_ = self._samples_ids_gen()
        for _, (
                xyz,
                rgb,
                category,
                instances,
        ) in self.iter_over_setups():
            for unique_category in category.unique():
                sample_id = next(id_)
                mask_c = category == unique_category
                xyz_c, rgb_c, cat_c, inst_c = (
                    xyz[mask_c],
                    rgb[mask_c],
                    category[mask_c],
                    instances[mask_c],
                )
                for unique_object in inst_c.unique():
                    mask_i = inst_c == unique_object
                    pt_cat = cat_c[mask_i].unique().size(0)
                    sample_xyz, sample_rgb = xyz_c[mask_i], rgb_c[mask_i]
                    assert (
                        pt_cat == 1), "Non-unique category for a single object"
                    data = Data(pos=sample_xyz, x=sample_rgb, cat=pt_cat)
                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    fs.torch_save(
                        data,
                        osp.join(self.processed_dir, f"sample_{sample_id}.pt"),
                    )


class InstanceSegmentationInLUT3D(InLUT3D):
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_instance_segmentation")

    @property
    def processed_file_names(self) -> List[str]:
        return [f"setup_{i}.pt" for i in range(self.SETUPS_NBR)]

    def collect_train_samples(self) -> List[str]:
        return [
            osp.join(self.processed_dir, f"{setup}.pt")
            for setup in self.train_setups
        ]

    def collect_test_samples(self) -> List[str]:
        return [
            osp.join(self.processed_dir, f"{setup}.pt")
            for setup in self.test_setups
        ]

    def process(self) -> None:
        for setup_name, (
                xyz,
                rgb,
                categories,
                instances,
        ) in self.iter_over_setups():
            data = Data(pos=xyz, x=rgb, cat=categories, instances=instances)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            fs.torch_save(data, osp.join(self.processed_dir,
                                         f"{setup_name}.pt"))


class SemanticSegmentationInLUT3D(InstanceSegmentationInLUT3D):
    SETUPS_NBR: int = 321
    SAMPLES_NBR: int = SETUPS_NBR

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_semantic_segmentation")

    def process(self) -> None:
        for setup_name, (xyz, rgb, categories, _) in self.iter_over_setups():
            data = Data(pos=xyz, x=rgb, cat=categories)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            fs.torch_save(data, osp.join(self.processed_dir,
                                         f"{setup_name}.pt"))
