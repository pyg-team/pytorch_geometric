import copy
import warnings
from typing import Any, List, Optional

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import hyper_subgraph, select


class HyperGraphData(Data):
    r"""A data object describing a hypergraph.
    The data object can hold node-level, link-level and graph-level attributes.
    This object differs from a standard data object by having hyperedges,
    i.e. edges that connect more than two nodes

    .. code-block:: python

        from torch_geometric.data import HyperGraphData

        x = torch.randn(4, 8)
        edge_index = torch.tensor([
                [0, 1, 0, 2, 1, 0, 1, 2, 3],
                [0, 0, 1, 1, 1, 2, 2, 2, 2]
            ])
        data = HyperGraphData(x=x, edge_index=edge_index, ...)

    Args:
        x (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Hyperedge list
            with shape :obj:`[2, num_nodes*num_edges_per_node]`.
            (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos, **kwargs)

    @property
    def num_edges(self) -> int:
        r"""Returns the number of hyperedges in the hypergraph.
        """
        return max(self.edge_index[1]) + 1

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`torch_geometric.loader.DataLoader`.

        Examples:
            >>> x = torch.randn(3, 16)
            >>> edge_index = torch.tensor([
            ...     [0, 1, 0, 2, 1],
            ...     [0, 0, 1, 1, 1]
            >>> ])
            >>> data = HyperGraphData(x = x, edge_index = edge_index)
            >>> data_list = [data, data]
            >>> loader = Dataloader(data_list, batch_size = 2)
            >>> batch = next(iter(loader))

            >>> batch.edge_index
            tensor([[0, 1, 0, 2, 1, 3, 4, 3, 5, 4],
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]])
        """

        if key == 'edge_index':
            return torch.tensor([[self.num_nodes], [self.num_edges]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def subgraph(self, subset: Tensor) -> 'HyperGraphData':
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        .. note::

            If only a subset of a hyperedge's connected nodes are to be
            selected in the subgraph, the hypergraph will remain in the
            subgraph, but only the selected nodes will be connected by
            the hyperedge. Hyperedges that only connects one node in the
            subgraph will be removed.

        Examples:
            >>> x = torch.randn(4, 16)
            >>> edge_index = torch.tensor([
            ...     [0, 1, 0, 2, 1, 1, 2, 4],
            ...     [0, 0, 1, 1, 1, 2, 2, 2]
            >>> ])
            >>> data = HyperGraphData(x = x, edge_index = edge_index)
            >>> subset = torch.tensor([1, 2, 4])
            >>> subgraph = data.subgraph(subset)
            >>> subgraph.edge_index
            tensor([[2, 1, 1, 2, 4],
            [0, 0, 1, 1, 1]])

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        out = hyper_subgraph(subset, self.edge_index, relabel_nodes=True,
                             num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        data = copy.copy(self)

        for key, value in self:
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                if subset.dtype == torch.bool:
                    data.num_nodes = int(subset.sum())
                else:
                    data.num_nodes = subset.size(0)
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, edge_mask, dim=cat_dim)

        return data

    def edge_subgraph(self, subset: Tensor) -> 'Data':
        raise NotImplementedError

    def to_heterogeneous(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[NodeType]] = None,
        edge_type_names: Optional[List[EdgeType]] = None,
    ):
        raise NotImplementedError

    def is_directed(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if 'edge_index' in self:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {self.edge_index.size()})",
                    raise_on_error)

        if 'edge_index' in self and self.edge_index.numel() > 0:
            if self.edge_index.min() < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min())})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index[0].max() >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max())})", raise_on_error)

        return status


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
