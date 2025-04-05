import warnings
from typing import Any, Dict, List

import torch
from torch import Tensor

from torch_geometric.typing import NodeType


def get_embeddings(
    model: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    from torch_geometric.nn import MessagePassing

    embeddings: List[Tensor] = []

    def hook(model: torch.nn.Module, inputs: Any, outputs: Any) -> None:
        # Clone output in case it will be later modified in-place:
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(outputs, Tensor)
        embeddings.append(outputs.clone())

    hook_handles = []
    for module in model.modules():  # Register forward hooks:
        if isinstance(module, MessagePassing):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    training = model.training
    model.eval()
    with torch.no_grad():
        model(*args, **kwargs)
    model.train(training)

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    return embeddings


def get_embeddings_hetero(
    model: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, List[Tensor]]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in a heterogeneous
    :obj:`model`, organized by edge type.

    Internally, this method registers forward hooks on all modules that process
    heterogeneous graphs in the model and runs the forward pass of the model.
    For heterogeneous models, the output is a dictionary where each key is a
    node type and each value is a list of embeddings from different layers.

    Args:
        model (torch.nn.Module): The heterogeneous GNN model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.

    Returns:
        Dict[NodeType, List[Tensor]]: A dictionary mapping each node type to
        a list of embeddings from different layers.
    """
    # Dictionary to store node embeddings by type
    node_embeddings_dict: Dict[NodeType, List[Tensor]] = {}

    # Hook function to capture node embeddings
    def hook(module, inputs, outputs):
        # Check if the outputs is a dictionary mapping node types to embeddings
        if isinstance(outputs, dict) and outputs:
            # Store embeddings for each node type
            for node_type, embedding in outputs.items():
                if node_type not in node_embeddings_dict:
                    node_embeddings_dict[node_type] = []
                node_embeddings_dict[node_type].append(embedding.clone())

    # List to store hook handles
    hook_handles = []

    # Find ModuleDict objects in the model
    for _, module in model.named_modules():
        submodules = list(module.children())
        submodules_contains_module_dict = any([
            isinstance(submodule, torch.nn.ModuleDict)
            for submodule in submodules
        ])
        if submodules_contains_module_dict:
            hook_handles.append(module.register_forward_hook(hook))

    # Run the model forward pass
    training = model.training
    model.eval()

    with torch.no_grad():
        model(*args, **kwargs)
    model.train(training)

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    return node_embeddings_dict
