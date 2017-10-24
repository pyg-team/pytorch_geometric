import torch.sparse as sparse

from .mm import mm
from .sum import sum


def SparseTensor(indices, values, size):
    return getattr(sparse, values.__class__.__name__)(indices, values, size)


__all__ = ['SparseTensor', 'mm', 'sum']
