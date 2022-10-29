from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.explain.base import ExplainerAlgorithm
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
        node_features_mask = torch.rand(x.shape)
        node_mask = torch.rand(x.shape[0])
        edge_mask = torch.rand(edge_index.shape[1])
        masks = (node_mask, edge_mask, node_features_mask)
        mask_keys = ("node_mask", "edge_mask", "node_features_mask")

        if edge_attr is not None:
            masks += (torch.rand(edge_attr.shape), )
            mask_keys += ("edge_features_mask", )

        return Explanation.from_masks(edge_index=edge_index, masks=masks,
                                      mask_keys=mask_keys, x=x,
                                      edge_attr=edge_attr)

    def explain(self, g: Data, model: torch.nn.Module,
                **kwargs) -> Explanation:
        return self.forward(x=g.x, edge_index=g.edge_index,
                            edge_attr=g.edge_attr, **kwargs)

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return mask_type != "layers"

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(y_hat, y)
