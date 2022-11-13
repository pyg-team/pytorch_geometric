from enum import Enum
from typing import Optional, Tuple, Union

import torch
from captum import attr
from torch import Tensor
from torch.nn.functional import mse_loss

from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
)
from torch_geometric.explain.explanations import Explanation
from torch_geometric.nn.models.explainer import to_captum


class CaptumMethod(Enum):
    Saliency = "Saliency"
    InputXGradient = 'InputXGradient'
    Deconvolution = 'Deconvolution'
    FeatureAblation = 'FeatureAblation'
    ShapleyValueSampling = 'ShapleyValueSampling'
    IntegratedGradients = 'IntegratedGradients'
    GradientShap = 'GradientShap'
    Occlusion = 'Occlusion'
    GuidedBackprop = 'GuidedBackprop'
    KernelShap = 'KernelShap'
    Lime = 'Lime'
    DeepLiftShap = 'DeepLiftShap'
    DeepLift = 'DeepLift'


class CaptumAlgorithm(ExplainerAlgorithm):
    """Wrapper for Captum.ai <https://captum.ai/>`_ attribution methods."""
    def __init__(self, captum_alg: Union[str, CaptumMethod], **kwargs) -> None:
        """Initialize the CaptumExplainer.

        Args:
            captum_alg (Union[str, CaptumMethod]): Captum algorithm to use.
            **kwargs: additional arguments for the Captum algorithm. E.g. for
                :obj:`Occlusion`, the sliding window shape.
        """
        super().__init__()
        self.captum_alg = CaptumMethod(captum_alg)
        self.captum_kwargs = kwargs

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        This method computes the loss to be used for the explanation algorithm.

        Args:
            y_hat (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            y (torch.Tensor): the reference output.
        """
        return mse_loss(y_hat, y, reduction="mean")

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        r"""Computes the explanation.

        Args:
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The input edge indices.
            explainer_config (ExplainerConfig): The explainer configuration.
            model_config (ModelConfig): the model configuration.
            target (torch.Tensor): the target of the model.
            target_index (int or torch.Tensor, optional): The target indices to
                explain. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """

        if not isinstance(target_index, int):
            raise ValueError(
                "CaptumExplainer only supports single target indices for now.")
        # check if the model was already used in the previous call
        mask_type = ""
        if explainer_config.node_mask_type is not None:
            mask_type += "node"
        if explainer_config.edge_mask_type is not None:
            if mask_type != "":
                mask_type += "_and_"
            mask_type += "edge"

        captum_model = to_captum(model, mask_type=mask_type,
                                 output_idx=target_index)
        captum_alg = getattr(attr, self.captum_alg.value)(captum_model)
        # compute the attribution for the inputs
        inputs, additional_forward_args = self._arange_inputs(
            x=x, edge_index=edge_index,
            node_mask_type=explainer_config.node_mask_type,
            edge_mask_type=explainer_config.edge_mask_type, **kwargs)
        attributions = captum_alg.attribute(**self._arange_args(
            input=inputs, target_idx=target_index,
            additional_forward_args=additional_forward_args))

        # build the explanation from the attribution
        if explainer_config.node_mask_type == MaskType.attributes:
            if explainer_config.edge_mask_type is None:
                node_mask = attributions.squeeze(0)
                edge_mask = None
            else:
                node_mask = attributions[0].squeeze(0)
                edge_mask = attributions[1]
        else:
            node_mask = None
            edge_mask = attributions

        return Explanation(
            x=x,
            edge_index=edge_index,
            node_features_mask=node_mask,
            edge_mask=edge_mask,
        )

    def _arange_inputs(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        node_mask_type: MaskType, edge_mask_type: MaskType, **kwargs
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[
            torch.Tensor, ...]]:
        """Arrange inputs for captum explainer.
        Create the main input and the additional forward arguments for the
        captum explainer based on the required type of explanations.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
           node_mask_type (MaskType): type of node mask.
            edge_mask_type (MaskType): type of edge mask.
            **kwargs: additional arguments for the forward function.

        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[
            torch.Tensor, ...]]: main input and additional forward arguments.
        """
        input_mask = torch.ones((1, edge_index.shape[1]), dtype=torch.float,
                                requires_grad=True)
        if node_mask_type == MaskType.attributes:
            if edge_mask_type is None:
                input = x.unsqueeze(0)
                additional_forward_args = (edge_index, )
            else:
                input = (x.unsqueeze(0), input_mask)
                additional_forward_args = (edge_index, )
        else:
            input = input_mask
            additional_forward_args = (
                x,
                edge_index,
            )

        for _, value in kwargs.items():
            additional_forward_args += (value, )
        return input, additional_forward_args

    def _arange_args(self, input, target_idx, additional_forward_args) -> dict:
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
            "target": target_idx,
            "additional_forward_args": additional_forward_args,
        }
        if self.captum_alg == CaptumMethod.IntegratedGradients:
            inputs["internal_batch_size"] = 1
        elif self.captum_alg == CaptumMethod.GradientShap:
            inputs["baselines"] = input
            inputs["n_samples"] = 1
        elif self.captum_alg in [
                CaptumMethod.DeepLiftShap, CaptumMethod.DeepLift
        ]:
            inputs["baselines"] = input

        # add the additional arguments from the constructor
        inputs.update(self.captum_kwargs)

        return inputs

    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> Tuple[bool, Optional[str]]:
        if explainer_config.edge_mask_type not in [None, MaskType.object]:
            return False, "CaptumExplainer only supports edge_mask_type of  \
                            none or object."

        if explainer_config.node_mask_type not in [MaskType.attributes, None]:
            return False, "CaptumExplainer only supports node_mask_type of  \
                            attributes or none."

        if explainer_config.explanation_type == ExplanationType.phenomenon:
            return False, "CaptumExplainer only supports explanation_type of  \
                            model."

        return True, None
