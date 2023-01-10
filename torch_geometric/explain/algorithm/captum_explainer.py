import logging
from typing import Optional, Union

import torch
from captum import attr  # noqa
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType


class CaptumExplainer(ExplainerAlgorithm):
    """Captum-based explainer algorithm.

    This explainer algorithm uses `Captum <https://captum.ai/>`_ to compute
    attributions.

    Args:
        attribution_method (Union[str, Attribution]): The Captum
            attribution method to use. Can be a string or a captum.attr method.
            Defaults to "IntegratedGradients".
        **kwargs: Additional arguments for the Captum attribution method.
    """
    def __init__(
        self,
        attribution_method: Union[str, Type],
        **kwargs,
    ):
        super().__init__()
        self.attribution_method = attribution_method
        self.kwargs = kwargs

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Optional[Tensor] = None,
        index: Optional[Tensor] = None,
        **kwargs,
    ) -> Explanation:

        return Explanation(node_mask=None, edge_mask=None)

    def supports(self) -> bool:
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in [None, MaskType.attributes]:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"'node_mask_type' in [None, 'attributes'], "
                          f"but got '{node_mask_type}'")
            return False

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type not in [None, MaskType.object]:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"'edge_mask_type' in [None, 'object'], "
                          f"but got '{edge_mask_type}'")
            return False

        return True
