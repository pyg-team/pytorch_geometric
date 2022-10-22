import torch

from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.explanations import Explanation
from torch_geometric.explainability.utils import CaptumModel


class CaptumExplainer(ExplainerAlgorithm):
    def __init__(self, method: str = "Saliency") -> None:
        super().__init__()
        self.method = method

    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        model: CaptumModel,
        mask_type: str = "node",
        **kwargs,
    ) -> Explanation:

        from captum import attr  # noqa

        # non optimal for now, but captum requires the model to instantiate
        # the explanation algorithm
        explainer = getattr(attr, self.method)(model)
        input_mask = torch.ones((1, edge_index.shape[1]), dtype=torch.float,
                                requires_grad=True)

        if mask_type == "node":
            input = x.clone().unsqueeze(0)
            additional_forward_args = (edge_index, )
        elif mask_type == "edge":
            input = input_mask
            additional_forward_args = (x, edge_index)
        elif mask_type == "node_and_edge":
            input = (x.clone().unsqueeze(0), input_mask)
            additional_forward_args = (edge_index, )

        if self.method == "IntegratedGradients":
            attributions = explainer.attribute(
                input,
                target=target.item(),
                internal_batch_size=1,
                additional_forward_args=additional_forward_args,
            )
        elif self.method == "GradientShap":
            attributions = explainer.attribute(
                input,
                target=target.item(),
                baselines=input,
                n_samples=1,
                additional_forward_args=additional_forward_args,
            )
        elif self.method in ["DeepLiftShap", "DeepLift"]:
            (attributions, ) = explainer.attribute(
                input,
                target=target.item(),
                baselines=input,
                additional_forward_args=additional_forward_args,
            )
        else:
            attributions = explainer.attribute(
                input,
                target=target.item(),
                additional_forward_args=additional_forward_args,
            )

        node_features_mask = None
        edge_mask = None
        if self.mask_type == "node":
            node_features_mask = attributions.squeeze(0)
        elif self.mask_type == "edge":
            edge_mask = attributions.squeeze(0)
        elif self.mask_type == "node_and_edge":
            node_features_mask = attributions[0].squeeze(0)
            edge_mask = attributions[1].squeeze(0)

        return Explanation(
            x=x,
            edge_index=edge_index,
            node_features_mask=node_features_mask,
            edge_mask=edge_mask,
            **kwargs,
        )
