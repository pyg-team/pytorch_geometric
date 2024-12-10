import torch
import torch.nn as nn


class ZINBLoss(nn.Module):
    """Custom loss function for the Zero-Inflated Negative Binomial (ZINB)
    distribution.

    Args:
        eps (float): A small constant to avoid division by zero or log(0).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        """Computes the negative log-likelihood of the ZINB distribution.

        Args:
            prediction (tuple): A tuple containing (mu, theta, pi) from the
                model:
                - mu: Mean of the Negative Binomial distribution.
                - theta: Dispersion parameter (greater than 0).
                - pi: Zero-inflation probability (between 0 and 1).
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: The computed ZINB loss.
        """
        mu, theta, pi = prediction
        return self.compute_zinb_loss(mu, theta, pi, target)

    def compute_zinb_loss(self, mu, theta, pi, target):
        """Computes the Zero-Inflated Negative Binomial loss components.

        Args:
            mu (torch.Tensor): Mean of the Negative Binomial distribution.
            theta (torch.Tensor): Dispersion parameter (greater than 0).
            pi (torch.Tensor): Zero-inflation probability (between 0 and 1).
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: The computed ZINB loss.
        """
        # Ensure valid values for stability
        mu = torch.clamp(mu, min=self.eps)
        theta = torch.clamp(theta, min=self.eps)
        pi = torch.clamp(pi, min=self.eps, max=1 - self.eps)
        target = torch.clamp(target, min=self.eps)

        # Log-likelihood of the Negative Binomial (NB) component
        log_nb = (torch.lgamma(theta + target) - torch.lgamma(target + 1) -
                  torch.lgamma(theta) + theta * torch.log(theta) +
                  target * torch.log(mu) -
                  (theta + target) * torch.log(theta + mu))

        # Log-likelihood of the zero-inflated component
        log_zero_inflated = torch.log(pi + (1 - pi) * torch.exp(log_nb))

        # Log-likelihood for non-zero values
        log_non_zero = torch.log(1 - pi) + log_nb

        # Combine likelihoods based on target values
        zinb_loss = torch.where(
            target < self.eps,  # If the target is zero
            -log_zero_inflated,  # Use zero-inflated likelihood
            -log_non_zero,  # Use regular NB likelihood
        )

        return zinb_loss.mean()  # Return the mean loss across all samples
