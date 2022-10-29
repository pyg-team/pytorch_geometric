from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.explainability.explanations import Explanation

from .base import ExplainerAlgorithm


class RandomExplainer(ExplainerAlgorithm):
    """Dummy explainer."""
    def explain(
        self,
        g: Data,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: Optional[int] = None,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Explanation:

        node_features_mask = torch.rand(g.x.shape)
        node_mask = torch.rand(g.x.shape[0])
        edge_mask = torch.rand(g.edge_index.shape[1])
        if "edge_attr" in g.keys:
            edge_features_mask = torch.rand(g.edge_attr.shape)
        return Explanation(
            x=g.x,
            edge_index=g.edge_index,
            node_features_mask=node_features_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_features_mask=edge_features_mask,
            **kwargs,
        )

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return mask_type != "layers"
