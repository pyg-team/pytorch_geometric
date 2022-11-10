from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from captum import attr
from torch.nn.functional import mse_loss

from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
)
from torch_geometric.explain.explanations import Explanation


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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        model: torch.nn.Module,
        target_index: Union[int, Tuple[int, ...], torch.Tensor,
                            List[Tuple[int, ...]], List[int]] = 0,
        node_mask_type: MaskType = MaskType.object,
        edge_mask_type: MaskType = None,
        **kwargs,
    ) -> Explanation:

        if not isinstance(target_index, int):
            raise ValueError(
                "CaptumExplainer only supports single target indices for now.")
        # check if the model was already used in the previous call

        captum_model = to_captum(model, edge_mask_type=edge_mask_type,
                                 node_mask_type=node_mask_type,
                                 output_idx=target_index)  # type: ignore
        captum_alg = getattr(attr, self.captum_alg.value)(captum_model)
        # compute the attribution for the inputs
        inputs, additional_forward_args = self._arange_inputs(
            x=x, edge_index=edge_index, node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type, **kwargs)
        attributions = captum_alg.attribute(**self._arange_args(
            input=inputs, target_idx=target_index,
            additional_forward_args=additional_forward_args))

        # build the explanation from the attribution
        if node_mask_type == MaskType.attributes:
            if edge_mask_type is None:
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
        explanation_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> Tuple[bool, Optional[str]]:
        if explanation_config.edge_mask_type not in [None, MaskType.object]:
            return False, "CaptumExplainer only supports edge_mask_type of  \
                            none or object."

        if explanation_config.node_mask_type not in [
                MaskType.attributes, None
        ]:
            return False, "CaptumExplainer only supports node_mask_type of  \
                            attributes or none."

        if explanation_config.explanation_type == ExplanationType.phenomenon:
            return False, "CaptumExplainer only supports explanation_type of  \
                            model."

        return True, None


class CaptumModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,
                 node_mask_type: Union[str, MaskType] = None,
                 edge_mask_type: Union[str, MaskType] = MaskType.object,
                 output_idx: Optional[int] = None):
        super().__init__()
        self.node_mask_type = MaskType(node_mask_type)
        self.edge_mask_type = MaskType(edge_mask_type)
        if self.node_mask_type is None:
            if self.edge_mask_type is None:
                raise ValueError(
                    "At least one mask type must be different from none.")
        self.model = model
        self.output_idx = output_idx

    def forward(self, mask, *args):
        """"""
        # The mask tensor, which comes from Captum's attribution methods,
        # contains the number of samples in dimension 0. Since we are
        # working with only one sample, we squeeze the tensors below.
        assert mask.shape[0] == 1, "Dimension 0 of input should be 1"

        if self.node_mask_type is None:  # just edge
            assert len(args) >= 2, "Expects at least x and edge_index as args."
        elif self.edge_mask_type is None:  # just node
            assert len(args) >= 1, "Expects at least edge_index as args."
        else:
            assert args[0].shape[0] == 1, "Dimension 0 of input should be 1"
            assert len(args[1:]) >= 1, "Expects at least edge_index as args."

        # Set edge mask:
        e_type = self.edge_mask_type
        n_type = self.node_mask_type
        if e_type == MaskType.object and n_type is None:
            set_masks(self.model, mask.squeeze(0), args[1],
                      apply_sigmoid=False)
        elif e_type == MaskType.object and n_type == MaskType.attributes:
            set_masks(self.model, args[0].squeeze(0), args[1],
                      apply_sigmoid=False)
            args = args[1:]

        if e_type == MaskType.object and n_type is None:
            x = self.model(*args)

        elif e_type is None and n_type == MaskType.attributes:
            x = self.model(mask.squeeze(0), *args)

        else:
            x = self.model(mask[0], *args)

        # Clear mask:
        if self.edge_mask_type == MaskType.object:
            clear_masks(self.model)

        if self.output_idx is not None:
            x = x[self.output_idx].unsqueeze(0)

        return x


def to_captum(model: torch.nn.Module,
              edge_mask_type: Union[str, MaskType] = MaskType.object,
              node_mask_type: Union[str, MaskType] = None,
              output_idx: Optional[int] = None) -> torch.nn.Module:
    r"""Converts a model to a model that can be used for
    `Captum.ai <https://captum.ai/>`_ attribution methods.

    .. code-block:: python

        from captum.attr import IntegratedGradients
        from torch_geometric.nn import GCN, to_captum

        model = GCN(...)
        ...  # Train the model.

        # Explain predictions for node `10`:
        output_idx = 10

        captum_model = to_captum(model, mask_type="edge",
                                 output_idx=output_idx)
        edge_mask = torch.ones(num_edges, requires_grad=True, device=device)

        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(edge_mask.unsqueeze(0),
                               target=int(y[output_idx]),
                               additional_forward_args=(x, edge_index),
                               internal_batch_size=1)

    .. note::
        For an example of using a Captum attribution method within PyG, see
        `examples/captum_explainability.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        captum_explainability.py>`_.

    Args:
        model (torch.nn.Module): The model to be explained.
        mask_type (str, optional): Denotes the type of mask to be created with
            a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
            and :obj:`"node_and_edge"`:

            1. :obj:`"edge"`: The inputs to the forward function should be an
               edge mask tensor of shape :obj:`[1, num_edges]`, a regular
               :obj:`x` matrix and a regular :obj:`edge_index` matrix.

            2. :obj:`"node"`: The inputs to the forward function should be a
               node feature tensor of shape :obj:`[1, num_nodes, num_features]`
               and a regular :obj:`edge_index` matrix.

            3. :obj:`"node_and_edge"`: The inputs to the forward function
               should be a node feature tensor of shape
               :obj:`[1, num_nodes, num_features]`, an edge mask tensor of
               shape :obj:`[1, num_edges]` and a regular :obj:`edge_index`
               matrix.

            For all mask types, additional arguments can be passed to the
            forward function as long as the first arguments are set as
            described. (default: :obj:`"edge"`)
        output_idx (int, optional): Index of the output element (node or link
            index) to be explained. With :obj:`output_idx` set, the forward
            function will return the output of the model for the element at
            the index specified. (default: :obj:`None`)
    """
    return CaptumModel(model, node_mask_type=node_mask_type,
                       edge_mask_type=edge_mask_type, output_idx=output_idx)
