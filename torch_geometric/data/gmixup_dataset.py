import copy
import os.path as osp
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import Tensor
from tqdm import tqdm

import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate
from torch_geometric.io import fs


class GMixupDataset(Dataset):
    r"""A dataset that applies G-Mixup data augmentation to its base dataset.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        TODO
    """

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)

        self.generate_graphons()
        
    def generate_graphons(self) -> None:
        self.graphons = [None] * len(self)
        
        for i in range(len(self)):
            data = self[i]
            # stuff
            self.graphons[i] = None
            
        return None

    def generate_graphs(self, idx: IndexType, mixing_param: float) -> 'Dataset':
        r"""Takes in a batch of graph label pairs and a mixing parameter λ, and returns the new synthetic graph(s) generated using G-Mixup
        """
        return None
    
