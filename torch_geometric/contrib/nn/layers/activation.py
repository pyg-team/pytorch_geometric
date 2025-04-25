import torch.nn as nn


def get_activation_function(activation: str = 'PReLU') -> nn.Module:
    r"""Returns the activation function specified by the string.
    The function supports the following activation functions:
    - ReLU
    - LeakyReLU
    - PReLU
    - tanh
    - SELU
    - ELU
    - Linear
    - GELU
    Args:
        activation (str): The name of the activation function to return.
            Default is 'PReLU'.

    Returns:
        nn.Module: The activation function.

    Raises:
        ValueError: If the activation function is not supported.
    """
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == "linear":
        return lambda x: x
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
