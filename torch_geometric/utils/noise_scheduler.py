import numpy as np


def get_smld_sigma_schedule(sigma_min: float, sigma_max: float,
                            num_scales: int) -> np.ndarray:
    r"""Generates a set of noise values on a logarithmic scale
    for "Score Matching with Langevin Dynamics" from the
    `"Generative Modeling by Estimating Gradients of the Data Distribution"
    <https://arxiv.org/abs/1907.05600>`_ paper.

    This function creates a numpy array of sigma values that define the
    schedule of noise levels used during Score Matching with Langevin Dynamics.
    The sigma values are determined on a logarithmic scale from `sigma_max`
    to `sigma_min`, inclusive.

    Args:
        sigma_min (float): The minimum value of sigma,
        corresponding to the lowest noise level.
        sigma_max (float): The maximum value of sigma,
        corresponding to the highest noise level.
        num_scales (int): The number of sigma values to generate,
        defining the granularity of the noise schedule.

    Returns:
        sigmas (np.ndarray): A numpy array of sigma values starting
        from `sigma_max` to `sigma_min` on a logarithmic scale.

    Example:
        >>> get_smld_sigma_schedule(0.01, 1.0, 10)
        array([1.        , 0.59948425, 0.35938137, 0.21544347, 0.12915497,
               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01      ])
    """
    sigmas = np.exp(
        np.linspace(np.log(sigma_max), np.log(sigma_min), num_scales))

    return sigmas


def get_diffusion_beta_schedule(schedule_type: str, beta_start: float,
                                beta_end: float,
                                num_diffusion_timesteps: int) -> np.ndarray:
    r"""Generates a schedule of beta values according to the specified
    strategy for the diffusion process from the
    `"Denoising Diffusion Probabilistic Models"
    <https://arxiv.org/abs/2006.11239>`_ paper.

    Beta values are used to scale the noise added during the diffusion
    process in generative models. This function creates an array of beta values
    according to a predefined schedule,which can be linear, quadratic,
    constant, inverse,or sigmoidal.

    Args:
        schedule_type (str): The type of schedule to use for beta values.
        Options include "quad", "linear", "const", or "sigmoid".
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        num_diffusion_timesteps (int): The number of timesteps for
        the diffusion process.

    Returns:
        np.ndarray: An array of beta values according to the
        specified schedule.

    Raises:
        NotImplementedError: If the schedule_type is not one of the predefined
        strategies.

    Examples:
        >>> get_diffusion_beta_schedule("linear", beta_start=0.1,
        beta_end=0.2, num_diffusion_timesteps=10)
        array([0.1, 0.111, ..., 0.2])
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if schedule_type == "quad":
        betas = (np.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        )**2)
    elif schedule_type == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps,
                            dtype=np.float64)
    elif schedule_type == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif schedule_type == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(schedule_type)
    return betas
