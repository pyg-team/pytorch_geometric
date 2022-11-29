from typing import List
import torch


# From https://github.com/mims-harvard/GraphXAI/
def perturb_node_features(x: torch.Tensor,
                          node_idx: int,
                          pert_feat: List[int] = [],
                          bin_dims: List[int] = [],
                          perturb_mode: str = 'gaussian',
                          device="cpu"):
    """

    Pick nodes with probability perturb_prob and perturb their features.
    The continuous dims are assumed to be non-negative.
    The discrete dims are required to be binary.
    Args:
        x (torch.Tensor, [n x d]): node features
        bin_dims (list of int): the list of binary dims
        perturb_mode (str):
            'scale': randomly scale (0-2) the continuous dim
            'gaussian': add a Gaussian noise to the continuous dim
            'uniform': add a uniform noise to the continuous dim
            'mean': set the continuous dims to their mean value
    Returns:
        x_pert (torch.Tensor, [n x d]): perturbed feature matrix
        node_mask (torch.Tensor, [n]): Boolean mask of perturbed nodes
    """

    cont_dims = [i for i in pert_feat if i not in bin_dims]
    len_cont = len(cont_dims)

    len_bin = len(bin_dims)
    scale_mul = 2
    x_pert = x.clone()

    max_val, _ = torch.max(x[:, cont_dims], dim=0, keepdim=True)

    if perturb_mode == 'scale':
        # Scale the continuous dims randomly
        x_pert[node_idx, cont_dims] *= scale_mul * torch.rand(cont_dims).to(device)
    elif perturb_mode == 'gaussian':
        # Add a Gaussian noise
        sigma = torch.std(x[:, cont_dims], dim=0, keepdim=True).squeeze(0)
        x_pert[node_idx, cont_dims] += sigma * torch.randn(len(cont_dims)).to(device)
    elif perturb_mode == 'uniform':
        # Add a uniform noise
        epsilon = 0.05 * max_val.squeeze(0)
        x_pert[node_idx, cont_dims] += 2 * epsilon * (torch.rand(len(cont_dims)) - 0.5)
    elif perturb_mode == 'mean':
        # Set to mean values
        mu = torch.mean(x[:, cont_dims], dim=0, keepdim=True).squeeze(0)
        x_pert[node_idx, cont_dims] = mu.to(device)
    else:
        raise ValueError("perturb_mode must be one of ['scale', 'gaussian', 'uniform', 'mean']")

    # Ensure feature value is between min_val and max_val
    min_val, _ = torch.min(x[:, cont_dims], dim=0, keepdim=True)
    x_pert[node_idx, cont_dims] = torch.max(torch.min(x_pert[node_idx, cont_dims], max_val), min_val)

    # Randomly flip the binary dims
    x_pert[node_idx, bin_dims] = torch.randint(2, size=(1, len_bin)).float().to(device)

    return x_pert[node_idx, pert_feat]