"""Annealed Langevin dynamics sampling, also called Langevin Monte Carlo.
This is used to sample score-based generative models, such as EDP-GNN.
The method iteratively samples from a series of noise-conditional score models
with varying noise levels.

TODO: do we need to specify the device/GPU?
TODO: do we need node_flags?
"""

import numpy as np
import torch
from torch import Tensor


def gen_init_sample(batch_size: int, num_nodes: int) -> Tensor:
    """Initialize a batch of adjacency matrices.

    The values are continuous and drawn from a folded normal distribution.
    Each element is drawn from N(0, 1) and then the absolute value is taken.
    The diagonal values are 0, so there are no self-loops.

    Args:
        batch_size (int): number of graphs (adjacency matrices)
        num_nodes (int): number of nodes in each graph

    Returns:
        adjs (Tensor): a batch of adjacency matrices with continuous values
    """
    # TODO: Add .to(self.dev) here if we need to specify the device/GPU:
    # TODO: is dtype=torch.float32 correct?
    adjs = torch.randn((batch_size, num_nodes, num_nodes),
                       dtype=torch.float32).triu(diagonal=1).abs()
    adjs = (adjs + adjs.transpose(-1, -2))
    return adjs


def sample(score_func: torch.nn.Module, adjs: Tensor, num_steps: int = 100,
           quantize=False) -> Tensor:
    """Run annealed Langevin dynamics sampling for one noise level.

    Args:
        score_func (torch.nn.Module): a noise conditional score-based
            generative model, TODO: change the type to EDP-GNN
        adjs (num_nodes): a batch of adjacency matrices
        num_steps (int): the number of iterations
        quantize (bool): whether to convert the adjacency matrices to discrete
            (binary) values. This should be True only for the final noise level

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    # Initialize constants
    """
    Algorithm 1 in https://arxiv.org/abs/2003.00638 uses alpha_i,
    but here grad_step_size=alpha_i/2.
    grad_step_size is the coefficient for the output of score_func.
    """
    grad_step_size = 1.0
    """
    From https://arxiv.org/abs/2003.00638:
    "We add another scaling coefficient epsilon_s, since it is a common
    practice of applying Langevin dynamics"
    """
    epsilon_s = 0.3
    """
    Noise is sampled from Gaussian, N(0, noise_variance)
    """
    noise_variance = epsilon_s * np.sqrt(grad_step_size * 2)

    for _ in range(num_steps):
        adjs = _sample_step(score_func, adjs, noise_variance)

    if quantize:
        # samples are in continuous space, but the graph edges are often
        # discrete, so we quantize to binary values
        # TODO: this can probably be optimized by not materializing matrices
        adjs = torch.where(adjs < 0.5, torch.zeros_like(adjs),
                           torch.ones_like(adjs))

    # TODO: do we need to detach from device?
    return adjs


def _sample_step(score_func: torch.nn.Module, adjs: Tensor,
                 noise_variance: float, grad_step_size: float) -> Tensor:
    """Run one step of annealed Langevin dynamics sampling.
    Sample symmetric Gaussian noise, and update the adjacency matrices based
    on the score function and noise.

    Args:
        score_func (torch.nn.Module): a noise conditional score-based
            generative model, TODO: change the type to EDP-GNN
        adjs (num_nodes): a batch of adjacency matrices
        noise_variance (float): variance of the Gaussian noise
        grad_step_size (float): step size coefficient for the score

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    # zeros on the diagonal, so no self-loops
    noise = torch.randn_like(adjs).triu(diagonal=1) * noise_variance
    noise_symmetric = noise + noise.transpose(-1, -2)
    score = score_func(adjs)

    new_adjs = adjs + grad_step_size * score + noise_symmetric
    return new_adjs
