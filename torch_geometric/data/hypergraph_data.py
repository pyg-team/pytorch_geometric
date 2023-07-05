import copy
import warnings
from typing import Any, List, Optional

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import hyper_subgraph, select


class HyperGraphData(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif key == 'face':
            return self.num_nodes
        elif key == 'edge_index':
            return torch.tensor([[self.num_nodes], [self.num_edges]])
        else:
            return 0

    def subgraph(self, subset: Tensor) -> 'HyperGraphData':
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
