import logging
from typing import Dict, List, Optional, Union

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
        is_hetero = isinstance(x, dict)

        if is_hetero:
            return self._forward_hetero(model, x, edge_index, target=target, 
                                       index=index, **kwargs)
        else:
            return self._forward_homo(model, x, edge_index, target=target,
                                     index=index, **kwargs)

    def _forward_homo(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
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
            if (isinstance(module, MessagePassing)
                    and module.explain is not False):
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

    def _forward_hetero(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        # Create a container for attention coefficients per edge type
        edge_type_to_alphas: Dict[EdgeType, List[Tensor]] = {}
        
        # Create a mapping from string format to tuple edge type
        str_to_edge_type = {}
        for edge_type in edge_index.keys():
            # Convert tuple edge type to string format
            assert len(edge_type) == 3  # (source, relation, target)
            source, relation, target = edge_type
            str_format = f"{source}__{relation}__{target}"
            str_to_edge_type[str_format] = edge_type
        
        # Create a hook wrapper function that captures edge_type information
        def create_hook_wrapper(maybe_edge_type_str: str):
            def hook_wrapper(module, msg_kwargs, out):
                # For string-based edge types, check if any part of the message kwargs matches our target
                edge_type = None
                if maybe_edge_type_str not in str_to_edge_type:
                    return
                edge_type = str_to_edge_type[maybe_edge_type_str]
                
                # Capture attention coefficients for the specific edge_type
                if edge_type not in edge_type_to_alphas:
                    edge_type_to_alphas[edge_type] = []
                    
                # Capture attention coefficients
                if 'alpha' in msg_kwargs[0]:
                    edge_type_to_alphas[edge_type].append(msg_kwargs[0]['alpha'].detach())
                elif getattr(module, '_alpha', None) is not None:
                    edge_type_to_alphas[edge_type].append(module._alpha.detach())
            
            return hook_wrapper

        # Register hooks for all MessagePassing modules in the model
        hook_handles = []
        for name, module in model.named_modules():
            if (isinstance(module, MessagePassing) and 
                module.explain is not False):
                
                # Extract edge type from the module name if it contains double underscore
                maybe_edge_type_str = None
                
                if '__' in name:
                    # Extract edge type from name patterns
                    parts = name.split('.')
                    for part in parts:
                        if '__' in part:
                            # Store the original string format for matching in hook
                            maybe_edge_type_str = part
                            break
                
                # Register a hook with the string representation
                # import pdb; pdb.set_trace()
                if maybe_edge_type_str is not None:
                    hook_fn = create_hook_wrapper(maybe_edge_type_str)
                    hook_handles.append(module.register_message_forward_hook(hook_fn))

        # Run the model forward pass to collect attention coefficients
        model(x, edge_index, **kwargs)

        # import pdb; pdb.set_trace()
        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # Check if we've collected any attention coefficients
        if not edge_type_to_alphas:
            raise ValueError("Could not collect any attention coefficients. "
                             "Please ensure that your model is using "
                             "attention-based GNN layers.")

        # Process attention coefficients for each edge type
        edge_masks_dict = {}
        for edge_type in edge_type_to_alphas.keys():
            alphas = edge_type_to_alphas[edge_type]
            if not alphas:
                continue
                
            for i, alpha in enumerate(alphas):
                # Ensure alpha doesn't exceed edge_index size
                alpha = alpha[:edge_index[edge_type].size(1)]
                
                # Reduce attention coefficients from multiple heads
                if alpha.dim() == 2:
                    alpha = getattr(torch, self.reduce)(alpha, dim=-1)
                    if isinstance(alpha, tuple):  # Handle torch.max output
                        alpha = alpha[0]
                elif alpha.dim() > 2:
                    raise ValueError(f"Can not reduce attention coefficients of "
                                 f"shape {list(alpha.size())}")
                alphas[i] = alpha
            
            # Combine attention coefficients across layers if we have any
            if len(alphas) > 1:
                alpha = torch.stack(alphas, dim=-1)
                alpha = getattr(torch, self.reduce)(alpha, dim=-1)
                if isinstance(alpha, tuple):
                    alpha = alpha[0]
            else:
                alpha = alphas[0]
            
            # Add the processed edge mask to the explanation
            edge_masks_dict[edge_type] = alpha

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
