from functools import lru_cache
from typing import Dict, List
import torch
from torch import Tensor
from torch_geometric.contrib.utils import Fiber
import torch.nn.functional as F

import e3nn.o3 as o3


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int, device) -> Tensor:
    """Get the (cached) Q^{d_out,d_in}_J matrices from equation (8)"""
    return o3.wigner_3j(J, d_in, d_out, dtype=torch.float64, device=device).permute(
        2, 1, 0
    )


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    all_degrees = list(range(2 * max_degree + 1))
    sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
    return torch.split(sh, [Fiber.degree_to_dim(d) for d in all_degrees], dim=1)


@torch.jit.script
def get_basis_script(
    max_degree: int,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
) -> Dict[str, Tensor]:
    """
    Compute pairwise bases matrices for degrees up to max_degree
    :param max_degree:            Maximum input or output degree
    :param use_pad_trick:         Pad some of the odd dimensions for a better use of Tensor Cores
    :param spherical_harmonics:   List of computed spherical harmonics
    :param clebsch_gordon:        List of computed CB-coefficients
    :param amp:                   When true, return bases in FP16 precision
    """
    basis = {}
    idx = 0
    # Double for loop instead of product() because of JIT script
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f"{d_in},{d_out}"
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(
                    torch.einsum(
                        "n f, k l f -> n l k",
                        spherical_harmonics[J].float(),
                        Q_J.float(),
                    )
                )

            basis[key] = torch.stack(K_Js, 2)  # Stack on second dim so order is n l f k
            idx += 1

    return basis


def get_basis(
    relative_pos: Tensor,
    max_degree: int = 4,
    compute_gradients: bool = False,
) -> Dict[str, Tensor]:
    spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    clebsch_gordon = get_all_clebsch_gordon(max_degree, relative_pos.device)

    with torch.autograd.set_grad_enabled(compute_gradients):
        basis = get_basis_script(
            max_degree=max_degree,
            spherical_harmonics=spherical_harmonics,
            clebsch_gordon=clebsch_gordon,
        )
        return basis
