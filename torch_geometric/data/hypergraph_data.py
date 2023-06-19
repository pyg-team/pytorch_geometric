import copy
from typing import Any

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import hyper_subgraph, select


class HyperGraphData(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, kwargs)

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
