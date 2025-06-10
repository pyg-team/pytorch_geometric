import logging
from typing import Dict, List, Optional, Union, overload

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.typing import EdgeType, NodeType


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
        self.is_hetero = False

    @overload
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
        ...

    @overload
    def forward(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        ...

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
        """Generate explanations based on attention coefficients."""
        self.is_hetero = isinstance(x, dict)

        # Collect attention coefficients
        alphas_dict = self._collect_attention_coefficients(
            model, x, edge_index, **kwargs)

        # Process attention coefficients
        if self.is_hetero:
            return self._create_hetero_explanation(model, alphas_dict,
                                                   edge_index, index, x)
        else:
            return self._create_homo_explanation(model, alphas_dict,
                                                 edge_index, index, x)

    @overload
    def _collect_attention_coefficients(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> List[Tensor]:
        ...

    @overload
    def _collect_attention_coefficients(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        **kwargs,
    ) -> Dict[EdgeType, List[Tensor]]:
        ...

    def _collect_attention_coefficients(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        **kwargs,
    ) -> Union[List[Tensor], Dict[EdgeType, List[Tensor]]]:
        """Collect attention coefficients from model layers."""
        if self.is_hetero:
            # For heterogeneous graphs, store alphas by edge type
            alphas_dict: Dict[EdgeType, List[Tensor]] = {}

            # Get list of edge types
            edge_types = list(edge_index.keys())

            # Hook function to capture attention coefficients by edge type
            def hook(module, msg_kwargs, out):
                # Find edge type from the module's full name
                module_name = getattr(module, '_name', None)
                if module_name is None:
                    return

                edge_type = None
                for edge_tuple in edge_types:
                    src_type, edge_name, dst_type = edge_tuple
                    # Check if all components appear in the module name in
                    # order
                    try:
                        src_idx = module_name.index(src_type)
                        edge_idx = module_name.index(edge_name, src_idx)
                        dst_idx = module_name.index(dst_type, edge_idx)
                        if src_idx < edge_idx < dst_idx:
                            edge_type = edge_tuple
                            break
                    except ValueError:  # Component not found
                        continue

                if edge_type is None:
                    return

                if edge_type not in alphas_dict:
                    alphas_dict[edge_type] = []

                # Extract alpha from message kwargs or module
                if 'alpha' in msg_kwargs[0]:
                    alphas_dict[edge_type].append(
                        msg_kwargs[0]['alpha'].detach())
                elif getattr(module, '_alpha', None) is not None:
                    alphas_dict[edge_type].append(module._alpha.detach())
        else:
            # For homogeneous graphs, store all alphas in a list
            alphas: List[Tensor] = []

            def hook(module, msg_kwargs, out):
                if 'alpha' in msg_kwargs[0]:
                    alphas.append(msg_kwargs[0]['alpha'].detach())
                elif getattr(module, '_alpha', None) is not None:
                    alphas.append(module._alpha.detach())

        # Register hooks for all message passing modules
        hook_handles = []
        for name, module in model.named_modules():
            if isinstance(module,
                          MessagePassing) and module.explain is not False:
                # Store name for hetero graph lookup in the hook
                if self.is_hetero:
                    module._name = name

                hook_handles.append(module.register_message_forward_hook(hook))

        # Forward pass to collect attention coefficients.
        model(x, edge_index, **kwargs)

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # Check if we collected any attention coefficients.
        if self.is_hetero:
            if not alphas_dict:
                raise ValueError(
                    "Could not collect any attention coefficients. "
                    "Please ensure that your model is using "
                    "attention-based GNN layers.")
            return alphas_dict
        else:
            if not alphas:
                raise ValueError(
                    "Could not collect any attention coefficients. "
                    "Please ensure that your model is using "
                    "attention-based GNN layers.")
            return alphas

    def _process_attention_coefficients(
        self,
        alphas: List[Tensor],
        edge_index_size: int,
    ) -> Tensor:
        """Process collected attention coefficients into a single mask."""
        for i, alpha in enumerate(alphas):
            # Ensure alpha doesn't exceed edge_index size
            alpha = alpha[:edge_index_size]

            # Reduce multi-head attention
            if alpha.dim() == 2:
                alpha = getattr(torch, self.reduce)(alpha, dim=-1)
                if isinstance(alpha, tuple):  # Handle torch.max output
                    alpha = alpha[0]
            elif alpha.dim() > 2:
                raise ValueError(f"Cannot reduce attention coefficients of "
                                 f"shape {list(alpha.size())}")
            alphas[i] = alpha

        # Combine attention coefficients across layers
        if len(alphas) > 1:
            alpha = torch.stack(alphas, dim=-1)
            alpha = getattr(torch, self.reduce)(alpha, dim=-1)
            if isinstance(alpha, tuple):  # Handle torch.max output
                alpha = alpha[0]
        else:
            alpha = alphas[0]

        return alpha

    def _create_homo_explanation(
        self,
        model: torch.nn.Module,
        alphas: List[Tensor],
        edge_index: Tensor,
        index: Optional[Union[int, Tensor]],
        x: Tensor,
    ) -> Explanation:
        """Create explanation for homogeneous graph."""
        # Get hard edge mask for node-level tasks
        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        # Process attention coefficients
        alpha = self._process_attention_coefficients(alphas,
                                                     edge_index.size(1))

        # Post-process mask with hard edge mask if needed
        alpha = self._post_process_mask(alpha, hard_edge_mask,
                                        apply_sigmoid=False)

        return Explanation(edge_mask=alpha)

    def _create_hetero_explanation(
        self,
        model: torch.nn.Module,
        alphas_dict: Dict[EdgeType, List[Tensor]],
        edge_index: Dict[EdgeType, Tensor],
        index: Optional[Union[int, Tensor]],
        x: Dict[NodeType, Tensor],
    ) -> HeteroExplanation:
        """Create explanation for heterogeneous graph."""
        edge_masks_dict = {}

        # Process each edge type separately
        for edge_type, alphas in alphas_dict.items():
            if not alphas:
                continue

            # Get hard edge mask for node-level tasks
            hard_edge_mask = None
            if self.model_config.task_level == ModelTaskLevel.node:
                src_type, _, dst_type = edge_type
                _, hard_edge_mask = self._get_hard_masks(
                    model, index, edge_index[edge_type],
                    num_nodes=max(x[src_type].size(0), x[dst_type].size(0)))

            # Process attention coefficients for this edge type
            alpha = self._process_attention_coefficients(
                alphas, edge_index[edge_type].size(1))

            # Apply hard mask if available
            edge_masks_dict[edge_type] = self._post_process_mask(
                alpha, hard_edge_mask, apply_sigmoid=False)

        # Create heterogeneous explanation
        explanation = HeteroExplanation()
        explanation.set_value_dict('edge_mask', edge_masks_dict)
        return explanation

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
