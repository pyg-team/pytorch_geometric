from typing import Dict, Any, Union, Callable

act_dict: Dict[str, Any] = {}
node_encoder_dict: Dict[str, Any] = {}
edge_encoder_dict: Dict[str, Any] = {}
stage_dict: Dict[str, Any] = {}
head_dict: Dict[str, Any] = {}
layer_dict: Dict[str, Any] = {}
pooling_dict: Dict[str, Any] = {}
network_dict: Dict[str, Any] = {}
config_dict: Dict[str, Any] = {}
loader_dict: Dict[str, Any] = {}
optimizer_dict: Dict[str, Any] = {}
scheduler_dict: Dict[str, Any] = {}
loss_dict: Dict[str, Any] = {}
train_dict: Dict[str, Any] = {}


def register_base(mapping: Dict[str, Any], key: str,
                  module: Any = None) -> Union[None, Callable]:
    r"""Base function for registering a module in GraphGym.

    Args:
        mapping (dict): Python dictionary to register the module.
            hosting all the registered modules
        key (string): The name of the module.
        module (any, optional): The module. If set to :obj:`None`, will return
            a decorator to register a module.
    """
    if module is not None:
        if key in mapping:
            raise KeyError(f"Module with '{key}' already defined")
        mapping[key] = module
        return

    # Other-wise, use it as a decorator:
    def wrapper(module):
        register_base(mapping, key, module)
        return module

    return wrapper


def register_act(key: str, module: Any = None):
    r"""Registers an activation function in GraphGym."""
    return register_base(act_dict, key, module)


def register_node_encoder(key: str, module: Any = None):
    r"""Registers a node feature encoder in GraphGym."""
    return register_base(node_encoder_dict, key, module)


def register_edge_encoder(key: str, module: Any = None):
    r"""Registers an edge feature encoder in GraphGym."""
    return register_base(edge_encoder_dict, key, module)


def register_stage(key: str, module: Any = None):
    r"""Registers a customized GNN stage in GraphGym."""
    return register_base(stage_dict, key, module)


def register_head(key: str, module: Any = None):
    r"""Registers a GNN prediction head in GraphGym."""
    return register_base(head_dict, key, module)


def register_layer(key: str, module: Any = None):
    r"""Registers a GNN layer in GraphGym."""
    return register_base(layer_dict, key, module)


def register_pooling(key: str, module: Any = None):
    r"""Registers a GNN global pooling/readout layer in GraphGym."""
    return register_base(pooling_dict, key, module)


def register_network(key: str, module: Any = None):
    r"""Registers a GNN model in GraphGym."""
    return register_base(network_dict, key, module)


def register_config(key: str, module: Any = None):
    r"""Registers a configuration group in GraphGym."""
    return register_base(config_dict, key, module)


def register_loader(key: str, module: Any = None):
    r"""Registers a data loader in GraphGym."""
    return register_base(loader_dict, key, module)


def register_optimizer(key: str, module: Any = None):
    r"""Registers an optimizer in GraphGym."""
    return register_base(optimizer_dict, key, module)


def register_scheduler(key: str, module: Any = None):
    r"""Registers a learning rate scheduler in GraphGym."""
    return register_base(scheduler_dict, key, module)


def register_loss(key: str, module: Any = None):
    r"""Registers a loss function in GraphGym."""
    return register_base(loss_dict, key, module)


def register_train(key: str, module: Any = None):
    r"""Registers a training function in GraphGym."""
    return register_base(train_dict, key, module)
