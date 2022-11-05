from typing import Optional

import torch

from torch_geometric.explain.base import ExplainerAlgorithm
from torch_geometric.explain.configuration import ExplainerConfig, ModelConfig
from torch_geometric.explain.explanations import Explanation


class RandomExplainer(ExplainerAlgorithm):
    """Dummy explainer."""
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                **kwargs) -> Explanation:
        """Returns a random explanation based on the shape of the inputs.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            edge_attr (torch.Tensor, optional): edge attributes.
                Defaults to None.

        Returns:
            Explanation: random explanation based on the shape of the inputs.
        """
        node_features_mask = torch.rand(x.shape, device=x.device)
        node_mask = torch.rand(x.shape[0], device=x.device)
        edge_mask = torch.rand(edge_index.shape[1], device=x.device)
        mask_dict = {
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "node_features_mask": node_features_mask,
        }

        if edge_attr is not None:
            mask_dict["edge_features_mask"] = torch.rand(
                edge_attr.shape, device=edge_attr.device)

        return Explanation(edge_index=edge_index, x=x, edge_attr=edge_attr,
                           **mask_dict)

    def supports(
        self,
        explanation_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        """Check if the explainer supports the user-defined settings.

        Returns true if the explainer supports the settings.


        Args:
            explanation_config (ExplainerConfig): the user-defined settings.
        """
        return True

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(y_hat, y)
