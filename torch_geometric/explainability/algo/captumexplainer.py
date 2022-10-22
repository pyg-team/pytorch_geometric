import torch

from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.explanations import Explanation
from torch_geometric.explainability.utils import CaptumModel


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
        input, additional_forward_args = self._arange_inputs(
            x, edge_index, mask_type, **kwargs)

        attributions = explainer.attribute(
            **self._arange_args(input, target, additional_forward_args))

        return self._create_explanation_from_masks(x, edge_index, attributions)

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return mask_type != "layers"

    def _arange_inputs(self, x, edge_index, mask_type, **kwargs):
        """Arrange inputs for captum explainer.

        Create the main input and the additional forward arguments for the
        captum explainer based on the required type of explanations.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            mask_type (str): type of explanation to be computed.

        Returns:
            Tuple[torch.Tensor,Tuple[torch.Tensor]]: main input and additional
                forward arguments.
        """
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
        # append additional kwargs given by user
        for _, value in kwargs.items():
            additional_forward_args += (value, )
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

    def _create_explanation_from_masks(self, x, edge_index, attributions):
        """Create explanation from masks.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices
            kwargs (_type_): _description_
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
            x=x,
            edge_index=edge_index,
            node_features_mask=node_features_mask,
            edge_mask=edge_mask,
        )
