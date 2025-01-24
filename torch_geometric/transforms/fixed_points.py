import math
import re

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('fixed_points')
class FixedPoints(BaseTransform):
    r"""Samples a fixed number of points and features from a point cloud
    (functional name: :obj:`fixed_points`).

    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples points
            without replacement. (default: :obj:`True`)
        allow_duplicates (bool, optional): In case :obj:`replace` is
            :obj`False` and :obj:`num` is greater than the number of points,
            this option determines whether to add duplicated nodes to the
            output points or not.
            In case :obj:`allow_duplicates` is :obj:`False`, the number of
            output points might be smaller than :obj:`num`.
            In case :obj:`allow_duplicates` is :obj:`True`, the number of
            duplicated points are kept to a minimum. (default: :obj:`False`)
    """
    def __init__(
        self,
        num: int,
        replace: bool = True,
        allow_duplicates: bool = False,
    ):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        assert num_nodes is not None

        if self.replace:
            choice = torch.from_numpy(
                np.random.choice(num_nodes, self.num, replace=True)).long()
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, value in data.items():
            if key == 'num_nodes':
                data.num_nodes = choice.size(0)
            elif bool(re.search('edge', key)):
                continue
            elif (isinstance(value, Tensor) and value.size(0) == num_nodes
                  and value.size(0) != 1):
                data[key] = value[choice]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'
