from collections import defaultdict
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType
from torch_geometric.typing import EdgeType, NodeType


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer that returns random explanations (useful for testing
    purposes).
    """
    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        edge_attr: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        assert isinstance(x, (Tensor, dict))

        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if isinstance(x, Tensor):  # Homogeneous graph.
            assert isinstance(edge_index, Tensor)

            node_mask = None
            if node_mask_type == MaskType.object:
                node_mask = torch.rand(x.size(0), 1, device=x.device)
            elif node_mask_type == MaskType.common_attributes:
                node_mask = torch.rand(1, x.size(1), device=x.device)
            elif node_mask_type == MaskType.attributes:
                node_mask = torch.rand_like(x)

            edge_mask = None
            if edge_mask_type == MaskType.object:
                edge_mask = torch.rand(edge_index.size(1), device=x.device)

            return Explanation(node_mask=node_mask, edge_mask=edge_mask)
        else:  # isinstance(x, dict):  # Heterogeneous graph.
            assert isinstance(edge_index, dict)

            node_dict = defaultdict(dict)
            for k, v in x.items():
                node_mask = None
                if node_mask_type == MaskType.object:
                    node_mask = torch.rand(v.size(0), 1, device=v.device)
                elif node_mask_type == MaskType.common_attributes:
                    node_mask = torch.rand(1, v.size(1), device=v.device)
                elif node_mask_type == MaskType.attributes:
                    node_mask = torch.rand_like(v)
                if node_mask is not None:
                    node_dict[k]['node_mask'] = node_mask

            edge_dict = defaultdict(dict)
            for k, v in edge_index.items():
                edge_mask = None
                if edge_mask_type == MaskType.object:
                    edge_mask = torch.rand(v.size(1), device=v.device)
                if edge_mask is not None:
                    edge_dict[k]['edge_mask'] = edge_mask

            return HeteroExplanation({**node_dict, **edge_dict})

    def supports(self) -> bool:
        return True
