import torch
from torch import Tensor


def get_characterization_score(fidelity: Tensor,
                               weights: Tensor = torch.tensor([1, 1])) \
        -> 'Tensor':
    r"""Returns the componentwise characterization score where fidelity[0]
    is the tensor containing F_{+} and fidelity[1] is the tensor
    containing F_{-}.

    The characterization score is calculated as
     .. math::
      charac = \frac{w_+ + w_-}{\frac{w_+}{fid_+}+\frac{w_-}{fid_-}}


    Args:
        weights (Tensor): Tensor containing the two weights of shape [2]
        fidelity (Tensor): Tensor containing fidelities of shape [2, n]

    Returns:
        Tensor: Tensor containing componentwise characterization
        score of shape [n]
    """
    components = torch.stack([fidelity[0], 1 - fidelity[1]])
    normed_weights = weights.div(weights.sum())

    return _weighted_harmonic_mean(normed_weights, components)


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
    """Computes the weighted harmonic mean of the two rows in components.
    Assumes the two weights are normalized.

    Args:
        weights (Tensor): Tensor containing the two weights of shape [2]
        components (Tensor): Tensor containing components of shape [2, n]

    Returns:
        Tensor: Tensor containing pairwise weighted harmonic means of shape [n]
    """
    if round(weights.sum().item(), 2) != 1:
        raise ValueError('weights must add up to 1')

    return torch.div(torch.prod(components, 0),
                     torch.flip(weights, [0]).matmul(components))
