"""Annealed Langevin dynamics sampling, also called Langevin Monte Carlo.
This is used to sample score-based generative models, such as EDP-GNN.
The method iteratively samples from a series of noise-conditional score models
with varying noise levels.
"""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def generate_initial_sample(
        batch_size: int, num_nodes: int,
        device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """Initialize a batch of adjacency matrices.

    The values are continuous and drawn from a folded normal distribution.
    Each element is drawn from N(0, 1) and then the absolute value is taken.
    The diagonal values are 0, so there are no self-loops.

    Args:
        batch_size (int): number of graphs (adjacency matrices)
        num_nodes (int): number of nodes in each graph
        device (Optional[torch.device]): torch.device

    Returns:
        adjs (Tensor): B x N x N, a batch of adjacency matrices with
            continuous values
        node_flags (Tensor): B x N, indicates whether each node exists
    """
    adjs = torch.randn((batch_size, num_nodes, num_nodes),
                       dtype=torch.float32).triu(diagonal=1).abs().to(device)
    adjs = (adjs + adjs.transpose(-1, -2))
    node_flags = adjs.sum(-1).squeeze(-1).gt(1e-5).to(torch.float32).to(device)
    return adjs, node_flags


def sample(score_func: Callable[[Tensor, Tensor], Tensor], adjs: Tensor,
           node_flags: Tensor, num_steps: int = 100, quantize: bool = False,
           grad_step_size: float = 1.0, epsilon_s: float = 0.3) -> Tensor:
    """Run annealed Langevin dynamics sampling for one noise level.

    Args:
        score_func (Callable[[Tensor, Tensor], Tensor]): a function that runs
            the noise conditional score-based generative model. The inputs are
            [adjs, node_flags]. The output is the score matrix.
        adjs (num_nodes): B x N x N, a batch of adjacency matrices
        node_flags (Tensor): B x N, indicates whether each node exists
        num_steps (int): the number of iterations
        quantize (bool): whether to convert the adjacency matrices to discrete
            (binary) values. This should be True only for the final noise level
        grad_step_size (float): coefficient applied to the output of score_func
            Algorithm 1 in https://arxiv.org/abs/2003.00638 uses alpha_i,
            but here grad_step_size=alpha_i/2.
        epsilon_s (float): coefficient applied to noise.
            From https://arxiv.org/abs/2003.00638: "We add another scaling
            coefficient epsilon_s, since it is a common practice of applying
            Langevin dynamics"

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    # Noise is sampled from normal distribtuion with variance noise_variance
    noise_variance = epsilon_s * np.sqrt(grad_step_size * 2)

    for _ in range(num_steps):
        adjs = _sample_step(score_func, adjs, node_flags, noise_variance,
                            grad_step_size)

    if quantize:
        # samples are in continuous space, but the graph edges are often
        # discrete, so we quantize to binary values
        adjs = torch.where(adjs < 0.5, 0, 1)
    adjs = adjs.detach()
    return adjs


def _sample_step(score_func: Callable[[Tensor, Tensor], Tensor], adjs: Tensor,
                 node_flags: Tensor, noise_variance: float,
                 grad_step_size: float) -> Tensor:
    """Run one step of annealed Langevin dynamics sampling.
    Sample symmetric Gaussian noise, and update the adjacency matrices based
    on the score function and noise.

    Args:
        score_func (Callable[[Tensor, Tensor], Tensor]): a function that runs
            the noise conditional score-based generative model. The inputs are
            [adjs, node_flags]. The output is the score matrix.
        adjs (num_nodes): a batch of adjacency matrices
        node_flags (Tensor): B x N, indicates whether each node exists
        noise_variance (float): variance of the Gaussian noise
        grad_step_size (float): step size coefficient for the score

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    # zeros on the diagonal, so no self-loops
    noise = torch.randn_like(adjs).triu(diagonal=1) * noise_variance
    noise_symmetric = noise + noise.transpose(-1, -2)
    adjs_with_noise = adjs + noise_symmetric

    score = score_func(adjs_with_noise, node_flags)
    new_adjs = adjs_with_noise + grad_step_size * score
    return new_adjs


def mask_adjs(adjs: Tensor, node_flags: Tensor) -> Tensor:
    """Use node_flags to mask out nodes in adjs that don't exist.

    Args:
        adjs (num_nodes): B x N x N, a batch of adjacency matrices
        node_flags (Tensor): B x N, indicates whether each node exists

    Returns:
        adjs (Tensor): a batch of adjacency matrices with values masked by
            node_flags
    """
    adjs = adjs * node_flags.unsqueeze(-1)
    adjs = adjs * node_flags.unsqueeze(-2)
    return adjs
