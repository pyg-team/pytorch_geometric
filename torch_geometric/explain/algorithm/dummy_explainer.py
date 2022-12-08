from typing import Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.typing import EdgeType, NodeType


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer for testing purposes."""
    def _forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Explanation:
        r"""Returns random explanations based on the shape of the inputs.

        Args:
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge attributes.
                (default: :obj:`None`)

        Returns:
            Explanation: A random explanation based on the shape of the inputs.
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        mask_dict = {}
        mask_dict['node_feat_mask'] = torch.rand_like(x)
        mask_dict['node_mask'] = torch.rand(num_nodes, device=x.device)
        mask_dict['edge_mask'] = torch.rand(num_edges, device=x.device)

        if edge_attr is not None:
            mask_dict['edge_feat_mask'] = torch.rand_like(edge_attr)

        return Explanation(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            **mask_dict,
        )

    def _forward_hetero(
        self,
        model: torch.nn.Module,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        r"""Returns random explanations based on the shape of the inputs.

        Args:
            model (torch.nn.Module): The model to explain.
            x_dict (Dict[NodeType, torch.Tensor]): The node features for each
                node type.
            edge_index_dict (Dict[EdgeType, torch.Tensor]): The edge indices
                for each edge type.
            edge_attr_dict (Dict[EdgeType, torch.Tensor], optional): The edge
                attributes for each edge type. (default: :obj:`None`)

        Returns:
            HeteroExplanation: A random explanation based on the shape of the
                inputs.
        """
        mask_dicts = {'node_feat_mask': {}, 'node_mask': {}, 'edge_mask': {}}
        for node_type, x in x_dict.items():
            mask_dicts['node_feat_mask'][node_type] = torch.rand_like(x)
            mask_dicts['node_mask'][node_type] = torch.rand(
                x.size(0), device=x.device)

        for edge_type, edge_index in edge_index_dict.items():
            mask_dicts['edge_mask'][edge_type] = torch.rand(
                edge_index.size(1), device=x.device)

            if edge_attr_dict is not None:
                mask_dicts['edge_feat_mask'][edge_type] = torch.rand_like(
                    edge_attr_dict[edge_type])

        return HeteroExplanation(x_dict=x_dict,
                                 edge_index_dict=edge_index_dict,
                                 edge_attr_dict=edge_attr_dict, **mask_dicts)

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        edge_attr: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Returns random explanations based on the shape of the inputs.

        Args:
            model (torch.nn.Module): The model to explain.
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The node
                features. This is a dictionary in the heterogeneous case.
            edge_index (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]):
                The edge indices. This is a dictionary in the heterogeneous
                case.
            edge_attr (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
                optional): The edge attributes. (default: :obj:`None`). This is
                a dictionary in the heterogeneous case.

        Returns:
            Union[Explanation, HeteroExplanation]: A random explanation based
                on the shape of the inputs.
        """
        if self._is_hetero:
            return self._forward_hetero(model, x, edge_index, edge_attr,
                                        **kwargs)
        return self._forward(model, x, edge_index, edge_attr, **kwargs)

    def supports(self) -> bool:
        return True
