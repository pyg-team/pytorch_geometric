import copy
import warnings
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import scatter

from .dense import HeteroLinear, Linear


class DummyLinear():
    def __init__(self, get_type: Union[NodeType, EdgeType],
                 types: Union[List[NodeType],
                              List[EdgeType]], hetero_module: torch.nn.Module):
        self.type = get_type
        self.types = types
        self.idx = types.index(get_type)
        self.in_channels = hetero_module.in_channels
        self.out_channels = hetero_module.out_channels
        self.hetero_weight = hetero_module.weight
        self.use_bias = hetero_module.kwargs.get('bias', True)

        self.hetero_bias = hetero_module.bias

    @property
    def weight(self):
        return self.hetero_weight[self.idx].view(self.in_channels,
                                                 self.out_channels)

    @property
    def bias(self):
        if self.use_bias:
            return self.hetero_bias[self.idx].view(self.out_channels)
        else:
            return None


class ToHeteroLinear(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        types: Union[List[NodeType], List[EdgeType]],
    ):
        super().__init__()

        self.types = types

        if isinstance(module, Linear):
            self.in_channels = module.in_channels
            self.out_channels = module.out_channels

        elif isinstance(module, torch.nn.Linear):
            self.in_channels = module.in_features
            self.out_channels = module.out_features

        else:
            raise ValueError(f"Expected 'Linear' module (got '{type(module)}'")

        self.bias = module.bias is not None

        # TODO: Need to handle `in_channels=-1` case.
        # TODO We currently assume that `x` is sorted according to `type`.
        self.hetero_module = HeteroLinear(
            self.in_channels,
            self.out_channels,
            num_types=len(types),
            is_sorted=True,
            bias=self.bias,
        )

    def __getitem__(
            self, get_type: Union[NodeType,
                                  EdgeType]) -> Union[Linear, DummyLinear]:
        # returns a Linear layer for type
        # neccesary to support the following examples:
        # 1) model.lin[node_type].weight.data = conv.root.data.t()
        # 2) model.lin[node_type].bias.data = conv.bias.data
        if not torch_geometric.typing.WITH_PYG_LIB:
            return self.hetero_module.lins[self.types.index(get_type)]
        else:
            return DummyLinear(get_type, self.types, self.hetero_module)

    def fused_forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        return self.hetero_module(x, type_vec)

    def dict_forward(
        self,
        x_dict: Dict[Union[NodeType, EdgeType], Tensor],
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:

        if not torch_geometric.typing.WITH_PYG_LIB:
            return {
                key: self.hetero_module.lins[i](x_dict[key])
                for i, key in enumerate(self.types)
            }

        x = torch.cat([x_dict[key] for key in self.types], dim=0)
        sizes = [x_dict[key].size(0) for key in self.types]
        type_vec = torch.arange(len(self.types), device=x.device)
        size = torch.tensor(sizes, device=x.device)
        type_vec = type_vec.repeat_interleave(size)
        outs = self.hetero_module(x, type_vec).split(sizes)
        return {key: out for key, out in zip(self.types, outs)}

    def forward(
        self,
        x: Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]],
        type_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]]:

        if isinstance(x, dict):
            return self.dict_forward(x)

        elif isinstance(x, Tensor) and type_vec is not None:
            return self.fused_forward(x, type_vec)

        raise ValueError(f"Encountered invalid forward types in "
                         f"'{self.__class__.__name__}'")


class ToHeteroMessagePassing(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        node_types: List[NodeType],
        edge_types: List[NodeType],
        aggr: str = 'sum',
    ):
        from torch_geometric.nn import HeteroConv, MessagePassing

        super().__init__()

        self.node_types = node_types
        self.node_type_to_index = {key: i for i, key in enumerate(node_types)}
        self.edge_types = edge_types

        if not isinstance(module, MessagePassing):
            raise ValueError(f"Expected 'MessagePassing' module "
                             f"(got '{type(module)}'")

        if (not hasattr(module, 'reset_parameters')
                and sum([p.numel() for p in module.parameters()]) > 0):
            warnings.warn(f"'{module}' will be duplicated, but its parameters "
                          f"cannot be reset. To suppress this warning, add a "
                          f"'reset_parameters()' method to '{module}'")

        convs = {edge_type: copy.deepcopy(module) for edge_type in edge_types}
        self.hetero_module = HeteroConv(convs, aggr)
        self.hetero_module.reset_parameters()

    def fused_forward(self, x: Tensor, edge_index: Tensor, node_type: Tensor,
                      edge_type: Tensor) -> Tensor:
        # TODO This currently does not fuse at all :(
        # TODO We currently assume that `x` and `edge_index` are both sorted
        # according to `type`.

        node_sizes = scatter(torch.ones_like(node_type), node_type, dim=0,
                             dim_size=len(self.node_types), reduce='sum')
        edge_sizes = scatter(torch.ones_like(edge_type), edge_type, dim=0,
                             dim_size=len(self.edge_types), reduce='sum')

        cumsum = torch.cat([node_type.new_zeros(1), node_sizes.cumsum(0)[:1]])

        xs = x.split(node_sizes.tolist())
        x_dict = {node_type: x for node_type, x in zip(self.node_types, xs)}

        # TODO Consider out-sourcing to its own function.
        edge_indices = edge_index.clone().split(edge_sizes.tolist(), dim=1)
        for (src, _, dst), index in zip(self.edge_types, edge_indices):
            index[0] -= cumsum[self.node_type_to_index[src]]
            index[1] -= cumsum[self.node_type_to_index[dst]]

        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in zip(self.edge_types, edge_indices)
        }

        out_dict = self.hetero_module(x_dict, edge_index_dict)
        return torch.cat([out_dict[key] for key in self.node_types], dim=0)

    def dict_forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        **kwargs,
    ) -> Dict[NodeType, Tensor]:
        return self.hetero_module(x_dict, edge_index_dict, **kwargs)

    def forward(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        node_type: OptTensor = None,
        edge_type: OptTensor = None,
        **kwargs,
    ) -> Union[Tensor, Dict[NodeType, Tensor]]:

        if isinstance(x, dict) and isinstance(edge_index, dict):
            return self.dict_forward(x, edge_index, **kwargs)

        elif (isinstance(x, Tensor) and isinstance(edge_index, Tensor)
              and node_type is not None and edge_type is not None):

            if len(kwargs) > 0:
                raise ValueError("Additional forward arguments not yet "
                                 "supported in fused mode")

            return self.fused_forward(x, edge_index, node_type, edge_type)

        raise ValueError(f"Encountered invalid forward types in "
                         f"'{self.__class__.__name__}'")
