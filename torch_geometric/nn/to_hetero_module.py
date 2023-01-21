import copy
import warnings
from typing import Dict, Optional, Union

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.typing import EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils.hetero import get_unused_node_types


class ToHeteroModule(torch.nn.Module):
    aggrs = {
        'sum': torch.add,
        # For 'mean' aggregation, we first sum up all feature matrices, and
        # divide by the number of matrices in a later step.
        'mean': torch.add,
        'max': torch.max,
        'min': torch.min,
        'mul': torch.mul,
    }

    def __init__(
        self,
        module: torch.nn.Module,
        metadata: Metadata,
        aggr: str = 'sum',
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.aggr = aggr
        assert len(metadata) == 2
        assert aggr in self.aggrs.keys()
        # check wether module is linear
        self.is_lin = isinstance(module, torch.nn.Linear) or isinstance(
            module, torch_geometric.nn.dense.Linear)
        # check metadata[0] has node types
        # check metadata[1] has edge types if module is MessagePassing
        assert len(metadata[0]) > 0 and (len(metadata[1]) > 0
                                         or not self.is_lin)
        if self.is_lin:
            # make HeteroLinear layer based on metadata
            if isinstance(module, torch.nn.Linear):
                in_ft = module.in_features
                out_ft = module.out_features
            else:
                in_ft = module.in_channels
                out_ft = module.out_channels
            heteromodule = torch_geometric.nn.dense.HeteroLinear(
                in_ft, out_ft,
                len(self.node_types)).to(list(module.parameters())[0].device)
            heteromodule.reset_parameters()
        else:
            # copy MessagePassing module for each edge type
            unused_node_types = get_unused_node_types(*metadata)
            if len(unused_node_types) > 0:
                warnings.warn(
                    f"There exist node types ({unused_node_types}) whose "
                    f"representations do not get updated during message "
                    f"passing as they do not occur as destination type in any "
                    f"edge type. This may lead to unexpected behaviour.")
            heteromodule = {}
            for edge_type in self.edge_types:
                heteromodule[edge_type] = copy.deepcopy(module)
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif sum([p.numel() for p in module.parameters()]) > 0:
                    warnings.warn(
                        f"'{module}' will be duplicated, but its parameters"
                        f"cannot be reset. To suppress this warning, add a"
                        f"'reset_parameters()' method to '{module}'")

        self.heteromodule = heteromodule

    def fused_forward(self, x: Tensor, edge_index: OptTensor = None,
                      node_type: OptTensor = None,
                      edge_type: OptTensor = None) -> Tensor:
        r"""
        Args:
            x: The input node features. :obj:`[num_nodes, in_channels]`
                node feature matrix.
            edge_index (LongTensor): The edge indices.
            node_type: The one-dimensional node type/index for each node in
                :obj:`x`.
            edge_type: The one-dimensional edge type/index for each edge in
                :obj:`edge_index`.
        """
        # (TODO) Add Sparse Tensor support
        if self.is_lin:
            # call HeteroLinear layer
            out = self.heteromodule(x, node_type)
        else:
            # iterate over each edge type
            for j, module in enumerate(self.heteromodule.values()):
                e_idx_type_j = edge_index[:, edge_type == j]
                o_j = module(x, e_idx_type_j)
                if j == 0:
                    out = o_j
                else:
                    out += o_j
        return out

    def dict_forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
        """
        # (TODO) Add Sparse Tensor support
        if self.is_lin:
            # fuse inputs
            x = torch.cat([x_j for x_j in x_dict.values()])
            size_list = [feat.shape[0] for feat in x_dict.values()]
            sizes = torch.tensor(size_list, dtype=torch.long, device=x.device)
            node_type = torch.arange(len(sizes), device=x.device)
            node_type = node_type.repeat_interleave(sizes)
            # HeteroLinear layer
            o = self.heteromodule(x, node_type)
            o_dict = {
                key: o_i.squeeze()
                for key, o_i in zip(x_dict.keys(), o.split(size_list))
            }
        else:
            o_dict = {}
            # iterate over each edge_type
            for j, (etype_j, module) in enumerate(self.heteromodule.items()):
                e_idx_type_j = edge_index_dict[etype_j]
                src_node_type_j = etype_j[0]
                dst_node_type_j = etype_j[-1]
                o_j = module(x_dict[src_node_type_j], e_idx_type_j)
                if dst_node_type_j not in o_dict.keys():
                    o_dict[dst_node_type_j] = o_j
                else:
                    o_dict[dst_node_type_j] += o_j
        return o_dict

    def forward(
        self,
        x: Union[Dict[NodeType, Tensor], Tensor],
        edge_index: Optional[Union[Dict[EdgeType, Tensor], Tensor]] = None,
        node_type: OptTensor = None,
        edge_type: OptTensor = None,
    ) -> Union[Dict[NodeType, Tensor], Tensor]:
        r"""
        Args:
            x (Dict[str, Tensor] or Tensor): A dictionary holding node feature
                information for each individual node type or the same
                features combined into one tensor.
            edge_index (Dict[Tuple[str, str, str], Tensor] or Tensor):
                A dictionary holding graph connectivity information for
                each individual edge type or the same values combined
                into one tensor.
            node_type: The one-dimensional relation type/index for each node in
                :obj:`x` if it is provided as a single tensor.
                Should be only :obj:`None` in case :obj:`x` is of type
                Dict[str, Tensor].
                (default: :obj:`None`)
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index` if it is provided as a single tensor.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                Dict[Tuple[str, str, str], Tensor].
                (default: :obj:`None`)
        """
        # check if x is passed as a dict or fused
        if isinstance(x, dict):
            # check what inputs to pass
            if self.is_lin:
                return self.dict_forward(x)
            else:
                if not isinstance(edge_index, dict):
                    raise TypeError("If x is provided as a dictionary, \
                        edge_index must be as well")
                return self.dict_forward(x, edge_index_dict=edge_index)
        else:
            if self.is_lin:
                if node_type is None:
                    raise ValueError('If x is a single tensor, \
                        node_type argument must be provided.')
                return self.fused_forward(x, node_type=node_type)
            else:
                if not isinstance(edge_index, Tensor):
                    raise TypeError("If x is provided as a Tensor, \
                        edge_index must be as well")
                if edge_type is None:
                    raise ValueError(
                        'If x and edge_indices are single tensors, \
                        node_type and edge_type arguments must be provided.')
                return self.fused_forward(x, edge_index=edge_index,
                                          edge_type=edge_type)
