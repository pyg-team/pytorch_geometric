from typing import Dict

import torch


class EdgeUpdater(torch.nn.Module):
    r"""An abstract base class for implementing custom edge update operators.
    TODO
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict:
        r"""
        Args: TODO
            index (torch.Tensor or SparseTensor): The indices of the edges on the graph.
            weight (torch.Tensor, optional): The weight values associated with each of the
                edges in :obj:`index`. If missing, then it is filled with ones.
            attr (torch.Tensor, optional): The attributes associated with each of the
                edges in :obj:`index`, unless :obj:`index` is a
                :class:`torch_sparse.SparseTensor`, in which case it should not be used.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

