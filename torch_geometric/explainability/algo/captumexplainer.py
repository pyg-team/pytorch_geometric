from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.algo.utils import CaptumModel
from torch_geometric.explainability.explanations import Explanation


class CaptumExplainer(ExplainerAlgorithm):
    """Wrapper for captum explainers.

    Wraper for captum explainers.

    Note:
        This algorithm expect the model to be callable with the following
        signature: model(x, edge_index, **kwargs).


    Args:
        method (str, optional): name of the captum explainer.
            Defaults to "Saliency".
    """
    def __init__(self, method: str = "Saliency") -> None:

        super().__init__()
        self.method = method

    def explain(
        self,
        g: Data,
        model: CaptumModel,
        target: torch.Tensor,
        target_index: Optional[int] = None,
        batch: Optional[torch.Tensor] = None,
        mask_type: str = "node",
        **kwargs,
    ) -> Explanation:

        from captum import attr  # noqa

        # non optimal for now, but captum requires the model to instantiate
        # the explanation algorithm
        explainer = getattr(attr, self.method)(model)
        input, additional_forward_args = self._arange_inputs(
            g, mask_type, **kwargs)

        attributions = explainer.attribute(
            **self._arange_args(input, target_index, additional_forward_args))

        return self._create_explanation_from_masks(g, attributions)

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return mask_type != "layers"

    def _arange_inputs(self, g, mask_type, **kwargs):
        """Arrange inputs for captum explainer.

        Create the main input and the additional forward arguments for the
        captum explainer based on the required type of explanations.

        Args:
            g (Data): input graph.
            mask_type (str): type of explanation to be computed.

        Returns:
            Tuple[torch.Tensor,Tuple[torch.Tensor]]: main input and additional
                forward arguments for the captum explainer.
        """
        input_mask = torch.ones((1, g.edge_index.shape[1]), dtype=torch.float,
                                requires_grad=True)

        additional_forward_args = None
        if mask_type == "node":
            input = g.x.clone().unsqueeze(0)
            additional_forward_args = (g.edge_index, )
        elif mask_type == "edge":
            input = input_mask
            additional_forward_args = (g.x, g.edge_index)
        elif mask_type == "node_and_edge":
            input = (g.x.clone().unsqueeze(0), input_mask)
            additional_forward_args = (g.edge_index, )

        if kwargs:
            if additional_forward_args is None:
                additional_forward_args = tuple(kwargs.values())
            else:
                additional_forward_args += tuple(kwargs.values())

        return input, additional_forward_args

    def _arange_args(self, input, target, additional_forward_args) -> dict:
        """Arrange inputs for captum explainer.

        Needed because of some inconsistencies in the captum API.

        Args:
            input (Tuple[Tensor ...]): input for which the attribution is
                computed.
            target (Tensor): output indices for which gradients are computed
            additional_forward_args (_type_): additional arguments for the
                forward function.

        Returns:
            dict: dictionary containing the inputs for the captum explainer.
        """
        inputs = {
            "inputs": input,
            "target": target.item(),
            "additional_forward_args": additional_forward_args,
        }
        if self.method == "IntegratedGradients":
            inputs["internal_batch_size"] = 1
        elif self.method == "GradientShap":
            inputs["baselines"] = input
            inputs["n_samples"] = 1
        elif self.method in ["DeepLiftShap", "DeepLift"]:
            inputs["baselines"] = input

        return inputs

    def _create_explanation_from_masks(self, g, attributions):
        """Create explanation from masks.

        Args:
            g (Data): input graph.
            attributions (Tuple[torch.Tensor]): masks returned by captum.
            kwargs (dict): additional information to store in the explanation.

        Returns:
            Explanation: explanation object.
        """
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
            x=g.x,
            edge_index=g.edge_index,
            node_features_mask=node_features_mask,
            edge_mask=edge_mask,
        )
