import inspect
import logging
import warnings
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.captum import (
    CaptumHeteroModel,
    CaptumModel,
    MaskLevelType,
    convert_captum_output,
    to_captum_input,
)
from torch_geometric.explain.config import MaskType, ModelMode
from torch_geometric.typing import EdgeType, NodeType


class CaptumExplainer(ExplainerAlgorithm):
    """A `Captum <https://captum.ai>`__-based explainer for identifying compact
    subgraph structures and node features that play a crucial role in the
    predictions made by a GNN.

    This explainer algorithm uses :captum:`null` `Captum <https://captum.ai/>`_
    to compute attributions.

    Currently, the following attribution methods are supported:

    * :class:`captum.attr.IntegratedGradients`
    * :class:`captum.attr.Saliency`
    * :class:`captum.attr.InputXGradient`
    * :class:`captum.attr.Deconvolution`
    * :class:`captum.attr.ShapleyValueSampling`
    * :class:`captum.attr.GuidedBackprop`

    Args:
        attribution_method (Attribution or str): The Captum attribution method
            to use. Can be a string or a :class:`captum.attr` method.
        **kwargs: Additional arguments for the Captum attribution method.
    """
    SUPPORTED_METHODS = [  # TODO: Add support for more methods.
        'IntegratedGradients',
        'Saliency',
        'InputXGradient',
        'Deconvolution',
        'ShapleyValueSampling',
        'GuidedBackprop',
    ]

    def __init__(
        self,
        attribution_method: Union[str, Any],
        **kwargs,
    ):
        super().__init__()

        import captum.attr  # noqa

        if isinstance(attribution_method, str):
            self.attribution_method = getattr(
                captum.attr,
                attribution_method,
            )
        else:
            self.attribution_method = attribution_method

        if not self._is_supported_attribution_method():
            raise ValueError(f"{self.__class__.__name__} does not support "
                             f"attribution method "
                             f"{self.attribution_method.__name__}")

        if kwargs.get('internal_batch_size', 1) != 1:
            warnings.warn("Overriding 'internal_batch_size' to 1")

        if 'internal_batch_size' in self._get_attribute_parameters():
            kwargs['internal_batch_size'] = 1

        self.kwargs = kwargs

    def _get_mask_type(self) -> MaskLevelType:
        r"""Based on the explainer config, return the mask type."""
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type
        if node_mask_type is not None and edge_mask_type is not None:
            mask_type = MaskLevelType.node_and_edge
        elif node_mask_type is not None:
            mask_type = MaskLevelType.node
        elif edge_mask_type is not None:
            mask_type = MaskLevelType.edge
        else:
            raise ValueError("Neither node mask type nor "
                             "edge mask type is specified.")
        return mask_type

    def _get_attribute_parameters(self) -> Dict[str, Any]:
        r"""Returns the attribute arguments."""
        signature = inspect.signature(self.attribution_method.attribute)
        return signature.parameters

    def _needs_baseline(self) -> bool:
        r"""Checks if the method needs a baseline."""
        parameters = self._get_attribute_parameters()
        if 'baselines' in parameters:
            param = parameters['baselines']
            if param.default is inspect.Parameter.empty:
                return True
        return False

    def _is_supported_attribution_method(self) -> bool:
        r"""Returns :obj:`True` if `self.attribution_method` is supported."""
        # This is redundant for now since all supported methods need a baseline
        if self._needs_baseline():
            return False
        elif self.attribution_method.__name__ in self.SUPPORTED_METHODS:
            return True
        return False

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:

        mask_type = self._get_mask_type()

        inputs, add_forward_args = to_captum_input(
            x,
            edge_index,
            mask_type,
            *kwargs.values(),
        )

        if isinstance(x, dict):
            metadata = (list(x.keys()), list(edge_index.keys()))
            captum_model = CaptumHeteroModel(
                model,
                mask_type,
                index,
                metadata,
            )
        else:
            metadata = None
            captum_model = CaptumModel(model, mask_type, index)

        self.attribution_method = self.attribution_method(captum_model)

        # In captum, the target is the index for which
        # the attribution is computed.
        if self.model_config.mode == ModelMode.regression:
            target = None
        else:
            target = target[index]

        attributions = self.attribution_method.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=add_forward_args,
            **self.kwargs,
        )

        node_mask, edge_mask = convert_captum_output(
            attributions,
            mask_type,
            metadata,
        )

        if not isinstance(x, dict):
            return Explanation(node_mask=node_mask, edge_mask=edge_mask)

        explanation = HeteroExplanation()
        explanation.set_value_dict('node_mask', node_mask)
        explanation.set_value_dict('edge_mask', edge_mask)
        return explanation

    def supports(self) -> bool:
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in [None, MaskType.attributes]:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"'node_mask_type' None or 'attributes' "
                          f"(got '{node_mask_type.value}')")
            return False

        # TODO (ramona): Confirm that output type is valid.
        return True
