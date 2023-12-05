import copy

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import scatter


class RECT_L(torch.nn.Module):
    r"""The RECT model, *i.e.* its supervised RECT-L part, from the
    `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`_ paper.
    In particular, a GCN model is trained that reconstructs semantic class
    knowledge.

    .. note::

        For an example of using RECT, see `examples/rect.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        rect.py>`_.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Intermediate size of each sample.
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        dropout (float, optional): The dropout probability.
            (default: :obj:`0.0`)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv = GCNConv(in_channels, hidden_channels, normalize=normalize)
        self.lin = Linear(hidden_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight.data)

    @torch.jit._overload_method
    def forward(self, x, edge_index, edge_weight=None):  # noqa: F811
        # type: (Tensor, Tensor, OptTensor) -> Tensor
        pass

    @torch.jit._overload_method
    def forward(self, x, edge_index, edge_weight=None):  # noqa: F811
        # type: (Tensor, SparseTensor, OptTensor) -> Tensor
        pass

    def forward(  # noqa: F811
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        x = self.conv(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    @torch.jit._overload_method
    def embed(self, x, edge_index, edge_weight=None):  # noqa: F811
        # type: (Tensor, Tensor, OptTensor) -> Tensor
        pass

    @torch.jit._overload_method
    def embed(self, x, edge_index, edge_weight=None):  # noqa: F811
        # type: (Tensor, SparseTensor, OptTensor) -> Tensor
        pass

    def embed(  # noqa: F811
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        with torch.no_grad():
            return self.conv(x, edge_index, edge_weight)

    def get_semantic_labels(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""Replaces the original labels by their class-centers."""
        with torch.no_grad():
            y = y[mask]
            mean = scatter(x[mask], y, dim=0, reduce='mean')
            return mean[y]

    def jittable(self, use_sparse_tensor: bool = False) -> torch.nn.Module:
        class EdgeIndexJittable(torch.nn.Module):
            def __init__(self, child: RECT_L):
                super().__init__()
                self.child = copy.deepcopy(child)
                self.child.conv = self.child.conv.jittable()

            def reset_parameters(self):
                self.child.reset_parameters()

            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: OptTensor = None,
            ) -> Tensor:
                return self.child(x, edge_index, edge_weight)

            @torch.jit.export
            def embed(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: OptTensor = None,
            ) -> Tensor:
                return self.child.embed(x, edge_index, edge_weight)

            @torch.jit.export
            def get_semantic_labels(
                self,
                x: Tensor,
                y: Tensor,
                mask: Tensor,
            ) -> Tensor:
                return self.child.get_semantic_labels(x, y, mask)

            def __repr__(self) -> str:
                return str(self.child)

        class SparseTensorJittable(torch.nn.Module):
            def __init__(self, child: RECT_L):
                super().__init__()
                self.child = copy.deepcopy(child)
                self.child.conv = self.child.conv.jittable()

            def reset_parameters(self):
                self.child.reset_parameters()

            def forward(
                self,
                x: Tensor,
                edge_index: SparseTensor,
                edge_weight: OptTensor = None,
            ):
                return self.child(x, edge_index, edge_weight)

            @torch.jit.export
            def embed(
                self,
                x: Tensor,
                edge_index: SparseTensor,
                edge_weight: OptTensor = None,
            ) -> Tensor:
                return self.child.embed(x, edge_index, edge_weight)

            @torch.jit.export
            def get_semantic_labels(
                self,
                x: Tensor,
                y: Tensor,
                mask: Tensor,
            ) -> Tensor:
                return self.child.get_semantic_labels(x, y, mask)

            def __repr__(self) -> str:
                return str(self.child)

        if use_sparse_tensor:
            return SparseTensorJittable(self)
        return EdgeIndexJittable(self)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels})')
