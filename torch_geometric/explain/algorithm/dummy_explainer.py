from typing import Optional

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer for testing purposes."""
    def forward(
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

    def supports(self) -> bool:
        return True
