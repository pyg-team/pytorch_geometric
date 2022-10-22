import torch

from torch_geometric.explainability.explanations import Explanation

from .base import ExplainerAlgorithm


class RandomExplainer(ExplainerAlgorithm):
    """Dummy explainer."""
    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        **kwargs,
    ) -> Explanation:

        node_features_mask = torch.rand(x.shape)
        node_mask = torch.rand(x.shape[0])
        edge_mask = torch.rand(edge_index.shape[1])
        if "edge_attr" in kwargs:
            edge_features_mask = torch.rand(kwargs["edge_attr"].shape)
        return Explanation(
            x=x,
            edge_index=edge_index,
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
