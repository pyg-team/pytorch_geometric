import logging
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType
from torch_geometric.typing import EdgeType, NodeType


class CaptumExplainer(ExplainerAlgorithm):
    """A Captum-based explainer for identifying compact subgraph structures and
    node features that play a crucial role in the predictions made by a GNN.

    This explainer algorithm uses `Captum <https://captum.ai/>`_ to compute
    attributions.

    Args:
        attribution_method (Attribution or str): The Captum attribution method
            to use. Can be a string or a :class:`captum.attr` method.
        **kwargs: Additional arguments for the Captum attribution method.
    """
    def __init__(
        self,
        attribution_method: Union[str, Any],
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
                          f"'node_mask_type' None or 'attributes' "
                          f"(got '{node_mask_type.value}')")
            return False

        return True
