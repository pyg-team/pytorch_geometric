import torch

if torch.cuda.is_available():
    from .compute_spline_basis import compute_spline_basis


def spline(input, kernel_size, is_open_spline, K, degree, basis_kernel):
    if input.is_cuda:
        return compute_spline_basis(input, kernel_size, is_open_spline, K, basis_kernel)
    else:
        raise NotImplementedError()
