from itertools import repeat
from typing import Sequence, Union

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_jitter')
class RandomJitter(BaseTransform):
    r"""Translates node positions by randomly sampled translation values
    within a given interval (functional name: :obj:`random_jitter`).
    In contrast to other random transformations,
    translation is applied separately at each position.

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """
    def __init__(
        self,
        translate: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:
        self.translate = translate

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        num_nodes, dim = data.pos.size()

        translate: Sequence[Union[float, int]]
        if isinstance(self.translate, (int, float)):
            translate = list(repeat(self.translate, times=dim))
        else:
            assert len(self.translate) == dim
            translate = self.translate

        jitter = data.pos.new_empty(num_nodes, dim)
        for d in range(dim):
            jitter[:, d].uniform_(-abs(translate[d]), abs(translate[d]))

        data.pos = data.pos + jitter

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'
