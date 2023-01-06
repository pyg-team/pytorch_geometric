import logging
from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel
from torch_geometric.nn.conv.message_passing import MessagePassing


class AttentionExplainer(ExplainerAlgorithm):
    r"""An explainer that uses the attention coefficients produced by an
    attention-based GNN (*e.g.*,
    :class:`~torch_geometric.nn.conv.GATConv`,
    :class:`~torch_geometric.nn.conv.GATv2Conv`, or
    :class:`~torch_geometric.nn.conv.TransformerConv`) as edge explanation.
    Attention scores across layers and heads will be aggregated according to
    the :obj:`reduce` argument.

    Args:
        reduce (str, optional): The method to reduce the attention scores
            across layers and heads. (default: :obj:`"max"`)
    """
    def __init__(self, reduce: str = 'max'):
        super().__init__()
        self.reduce = reduce

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.model:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"model explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            # We need to compute the hard edge mask to properly clean up edge
            # attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        alphas: List[Tensor] = []

        def hook(module, msg_kwargs, out):
            if 'alpha' in msg_kwargs[0]:
                alphas.append(msg_kwargs[0]['alpha'].detach())
            elif getattr(module, '_alpha', None) is not None:
                alphas.append(module._alpha.detach())

        hook_handles = []
        for module in model.modules():  # Register message forward hooks:
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_message_forward_hook(hook))

        model(x, edge_index, **kwargs)

        for handle in hook_handles:  # Remove hooks:
            handle.remove()

        if len(alphas) == 0:
            raise ValueError("Could not collect any attention coefficients. "
                             "Please ensure that your model is using "
                             "attention-based GNN layers.")

        for i, alpha in enumerate(alphas):
            alpha = alpha[:edge_index.size(1)]  # Respect potential self-loops.
            if alpha.dim() == 2:
                alpha = getattr(torch, self.reduce)(alpha, dim=-1)
                if isinstance(alpha, tuple):  # Respect `torch.max`:
                    alpha = alpha[0]
            elif alpha.dim() > 2:
                raise ValueError(f"Can not reduce attention coefficients of "
                                 f"shape {list(alpha.size())}")
            alphas[i] = alpha

        if len(alphas) > 1:
            alpha = torch.stack(alphas, dim=-1)
            alpha = getattr(torch, self.reduce)(alpha, dim=-1)
            if isinstance(alpha, tuple):  # Respect `torch.max`:
                alpha = alpha[0]
        else:
            alpha = alphas[0]

        alpha = self._post_process_mask(alpha, hard_edge_mask,
                                        apply_sigmoid=False)

        return Explanation(edge_mask=alpha)
