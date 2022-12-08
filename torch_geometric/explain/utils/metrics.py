import torch
from torch import Tensor


def get_characterization_score(weights: Tensor,
                               fidelities: Tensor) -> 'Tensor':
    r"""Returns the componentwise characterization score of
    fidelities[0] and fidelities[1]


    Args:
        weights (Tensor): Tensor containing the two weights
        fidelities (Tensor): Tensor containing fidelities of shape [2, n]

    Returns:
        Tensor: componentwise characterization score
    """
    components = torch.stack([fidelities[0], 1 - fidelities[1]])

    return _weighted_harmonic_mean(weights, components)


def get_fidelity_curve_auc(fidelities: Tensor, x: Tensor) -> 'Tensor':
    r"""Returns the AUC for the fidelity curve
    More precisely returns the AUC for:

     .. math::
        y = \frac{Fid_{+}}{1 - Fid_{-}}

    Expects the input to be sorted, i.e. x should be an ascending sequence.

    Args:
        fidelities (Tensor): Tensor containing fidelities
        x (Tensor): Tensor containing x-axis


    Returns:
        Tensor: area under the fidelity curve
    """

    if torch.any(fidelities[1] == 1):
        raise ValueError('0 division when dividing by 1 - fidelities[1].')

    y = torch.div(fidelities[0], 1 - fidelities[1])
    area = _auc(x, y)

    return area


def _auc(x: Tensor, y: Tensor) -> 'Tensor':

    if torch.any(torch.diff(x) < 0):
        raise ValueError('x must be an increasing sequence')

    area = torch.trapezoid(y, x)

    return area


def _weighted_harmonic_mean(weights: Tensor, components: Tensor) -> 'Tensor':

    return torch.div(torch.prod(components, 0),
                     torch.flip(weights, [0]).matmul(components))
