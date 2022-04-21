from typing import Optional, Union

from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('add_positional_encoding')
class AddPositionalEncoding(BaseTransform):
    r"""Adds positional encoding to the given graph (functional name:
    :obj:`add_positional_encoding`).

    Args:
        name: (str): The type of the positional encoding
            (:obj:`"laplacian_eigenvector_pe"`, :obj:`"random_walk_pe"`).
            If set to :obj:`"laplacian_eigenvector_pe"`, the Laplacian
            eigenvector positional encoding from the `"Benchmarking
            Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
            paper.
            If set to :obj:`"random_walk_pe"`, the random walk positional
            encoding from the `"Graph Neural Networks with Learnable
            Structural and Positional Representations"
            <https://arxiv.org/abs/2110.07875>`_ paper.
        num_channels: (int): The number of channels. If :attr:`name` is
            :obj:`"laplacian_eigenvector_pe"`, it will be the number of the
            smallest non-trivial eigenvectors. If :attr:`name` is
            :obj:`"random_walk_pe"`, it will be the number of the random walk
            steps.
        method: (str, optional): The method to add positional encoding.
            If set to :obj:`"attr"`, a new attribute will be added to the
                data object.
            If set to :obj:`"cat"`, will be concatenated to the existing
                :obj:`x` in the data object. (default: :obj:`attr`)
    """
    def __init__(self, name: str, num_channels: int, method: Optional[str] = 'attr'):
        self.name = name
        self.num_channels = num_channels
        self.method = method
        assert name in ['laplacian_eigenvector_pe', 'random_walk_pe']
        assert method in ['attr', 'cat']

    def __call__(self, data: Data):

        if self.name == 'laplacian_eigenvector_pe':
            pass
        elif self.name == 'random_walk_pe':
            pass

        if self.method == 'attr':
            pass
        elif self.method == 'cat':
            pass

        return data
