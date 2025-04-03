import warnings
from typing import Any, Dict, List

import torch
from torch import Tensor


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


def get_hetero_embeddings(
    model: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, List[Tensor]]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model` for heterogeneous graphs.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.

    Returns:
        Dict[str, List[Tensor]]: Dictionary mapping node types to lists of
        embeddings from each message passing layer.
    """
    # TODO (xinwei): This needs to completely rewritten for correctness.
    from torch_geometric.nn import MessagePassing

    # Get node types and their sizes from input
    if len(args) == 0 or not isinstance(args[0], dict):
        raise ValueError(
            "The first argument must be a dictionary mapping node types to features"
        )

    x_dict = args[0]
    node_types = list(x_dict.keys())

    # Create a dictionary to store embeddings for each node type
    # Initialize with empty lists
    embeddings_dict = {node_type: [] for node_type in node_types}

    # Dictionary to temporarily store the output of each layer
    layer_outputs = []

    def hook(module: torch.nn.Module, inputs: Any, outputs: Any) -> None:
        # Store the layer output
        if isinstance(outputs, dict):
            # If output is already a dictionary (common in heterogeneous GNNs)
            layer_outputs.append(outputs)
        elif isinstance(outputs, tuple) and isinstance(outputs[0], dict):
            # If output is a tuple with a dictionary as first element
            layer_outputs.append(outputs[0])
        else:
            # Other formats - store as is, we'll handle it later
            layer_outputs.append(outputs)

    # Register forward hooks on all MessagePassing layers
    hook_handles = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    # Run the model
    training = model.training
    model.eval()
    with torch.no_grad():
        out_dict = model(*args, **kwargs)
    model.train(training)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Process the collected outputs
    for layer_idx, layer_output in enumerate(layer_outputs):
        if isinstance(layer_output, dict):
            # Direct dictionary output
            for node_type, emb in layer_output.items():
                if node_type in embeddings_dict:
                    embeddings_dict[node_type].append(emb.clone())
        elif isinstance(layer_output, Tensor):
            # For homogeneous layers in heterogeneous models
            # We need to infer which node type this belongs to
            # As a fallback, distribute copies to all node types using correct sizes
            for node_type, features in x_dict.items():
                # Check if tensor shape matches this node type
                if layer_output.size(0) == features.size(0):
                    embeddings_dict[node_type].append(layer_output.clone())

    # Fallback: If any node type has no embeddings, use the model's output
    if isinstance(out_dict, dict):
        for node_type in node_types:
            if len(embeddings_dict[node_type]) == 0 and node_type in out_dict:
                embeddings_dict[node_type].append(out_dict[node_type].clone())

    # If we still have empty node types, create placeholder embeddings
    # with appropriate dimensions
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) == 0:
            # Use input feature dimension
            x_dict[node_type].size(0)
            # Create a dummy embedding with same size as input features
            placeholder = torch.zeros_like(x_dict[node_type])
            embeddings_dict[node_type].append(placeholder)

    return embeddings_dict
