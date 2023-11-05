import numbers
from itertools import repeat
from typing import Sequence, Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_jitter')
class RandomJitter(BaseTransform):
    r"""Translates node positions by randomly sampled translation values
    within a given interval (functional name: :obj:`random_jitter`).
    In contrast to other random transformations,
    translation is applied separately at each position

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """
    def __init__(self, translate: Union[float, int, Sequence]):
        self.translate = translate

    def forward(self, data: Data) -> Data:
        (n, dim), t = data.pos.size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.pos.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        data.pos = data.pos + torch.stack(ts, dim=-1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'
