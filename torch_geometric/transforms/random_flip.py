import random

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_flip')
class RandomFlip(BaseTransform):
    """Flips node positions along a given axis randomly with a given
    probability (functional name: :obj:`random_flip`).

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """
    def __init__(self, axis: int, p: float = 0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data: Data) -> Data:
        if random.random() < self.p:
            pos = data.pos.clone()
            pos[..., self.axis] = -pos[..., self.axis]
            data.pos = pos
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'
