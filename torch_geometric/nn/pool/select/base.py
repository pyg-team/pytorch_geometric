from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils.mixin import CastMixin


@dataclass(init=False)
class SelectOutput(CastMixin):
    r"""The output of the :class:`Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index (torch.Tensor): The indices of the selected nodes.
        cluster_index (torch.Tensor): The indices of the clusters each node in
            :obj:`node_index` is assigned to.
        num_clusters (int): The number of clusters.
        weight (torch.Tensor, optional): A weight vector, denoting the strength
            of the assignment of a node to its cluster. (default: :obj:`None`)
    """
    node_index: Tensor
    cluster_index: Tensor
    num_clusters: int
    weight: Optional[Tensor] = None

    def __init__(
        self,
        node_index: Tensor,
        cluster_index: Tensor,
        num_clusters: int,
        weight: Optional[Tensor] = None,
    ):
        if node_index.dim() != 1:
            raise ValueError(f"Expected 'node_index' to be one-dimensional "
                             f"(got {node_index.dim()} dimensions)")

        if cluster_index.dim() != 1:
            raise ValueError(f"Expected 'cluster_index' to be one-dimensional "
                             f"(got {cluster_index.dim()} dimensions)")

        if node_index.numel() != cluster_index.numel():
            raise ValueError(f"Expected 'node_index' and 'cluster_index' to "
                             f"hold the same number of values (got "
                             f"{node_index.numel()} and "
                             f"{cluster_index.numel()} values)")

        if weight is not None and weight.dim() != 1:
            raise ValueError(f"Expected 'weight' vector to be one-dimensional "
                             f"(got {weight.dim()} dimensions)")

        if weight is not None and weight.numel() != node_index.numel():
            raise ValueError(f"Expected 'weight' to hold {node_index.numel()} "
                             f"values (got {weight.numel()} values)")

        self.node_index = node_index
        self.cluster_index = cluster_index
        self.num_clusters = num_clusters
        self.weight = weight


class Select(torch.nn.Module):
    r"""An abstract base class for implementing custom node selections as
    described in the `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper, which maps the nodes of an
    input graph to supernodes in the coarsened graph.

    Specifically, :class:`Select` returns a :class:`SelectOutput` output, which
    holds a (sparse) mapping :math:`\mathbf{C} \in {[0, 1]}^{N \times C}` that
    assigns selected nodes to one or more of :math:`C` super nodes.
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self, *args, **kwargs) -> SelectOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
