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
    """Initialize adjacency matrices.

    The values are continuous and drawn from a folded normal distribution.
    Each element is drawn from N(0, 1) and then the absolute value is taken.
    The diagonal values are 0 (no self-loops)

    Args:
        batch_size (int): number of graphs (adjacency matrices)
        num_nodes (int): number of nodes in each graph

    Returns:
        adjs_c (Tensor): a batch of adjacency matrices with continuous values
    """
    # Add .to(self.dev) here if we need to specify the device/GPU:
    adjs_c = torch.randn((batch_size, num_nodes, num_nodes),
                         dtype=torch.float32).triu(diagonal=1).abs()
    adjs_c = (adjs_c + adjs_c.transpose(-1, -2))
    return adjs_c


def sample(score_func: torch.nn.Module, adjs: Tensor, num_steps: int = 100,
           quantize=False) -> Tensor:
    """Run annealed Langevin dynamics sampling for one noise level.

    TODO: documentation

    Args:
        score_func (torch.nn.Module): a noise conditional score-based
            generative model
            TODO: change the type to EDP-GNN or something else
        adjs (num_nodes): a batch of adjacency matrices
        num_steps (int): the number of iterations
        quantize (bool): whether to convert the adjacency matrices to discrete
            (binary) values. This should be True only for the final noise level

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    """
    From the other github repo: "lambda in langevin dynamics"
    The paper uses alpha_i, but here grad_step_size=alpha_i/2
    grad_step_size is the coefficient for the score_func output
    """
    grad_step_size = 1.0
    """
    According to the paper:
    "We add another scaling coefficient epsilon_s, since it is a common
    practice of applying Langevin dynamics"
    """
    epsilon_s = 0.3
    """
    noise_variance is epsilon_s * sqrt(alpha_i) from the paper
    Noise is sampled from a Gaussian, N(mean=0, variance=noise_variance)
    """
    noise_variance = epsilon_s * np.sqrt(grad_step_size * 2)

    for _ in range(num_steps):
        adjs = _step_sample(score_func, adjs, noise_variance)

    if quantize:
        # samples are in continuous space, but the graph edges are often
        # discrete, so we quantize to binary values
        adjs = torch.where(adjs < 0.5, torch.zeros_like(adjs),
                           torch.ones_like(adjs))

    # TODO: detach?
    return adjs


def _step_sample(score_func: torch.nn.Module, adjs: Tensor,
                 noise_variance: float, grad_step_size: float) -> Tensor:
    """Run one step of annealed Langevin dynamics sampling.
    Sample symmetric Gaussian noise, and update the adjacency matrices based
    on the score function and noise.

    Args:
        score_func (torch.nn.Module): a noise conditional score-based
            generative model
            TODO: change the type to EDP-GNN or something else
        adjs (num_nodes): a batch of adjacency matrices
        noise_variance (float): variance of the Gaussian noise
        grad_step_size (float): step size coefficient for the score

    Returns:
        adjs (Tensor): a batch of adjacency matrices
    """
    noise = torch.randn_like(adjs).triu(diagonal=1) * noise_variance
    noise_symmetric = noise + noise.transpose(-1, -2)
    score = score_func(adjs)

    adjs = adjs + grad_step_size * score + noise_symmetric
    return adjs
