import torch.sparse as sparse

from .mm import mm
from .mm_diagonal import mm_diagonal
from .sum import sum
from .eye import eye
from .stack import stack


def SparseTensor(indices, values, size):
    return getattr(sparse, values.__class__.__name__)(indices, values, size)


__all__ = ['SparseTensor', 'mm', 'mm_diagonal', 'sum', 'eye', 'stack']
