import torch

if torch.cuda.is_available():
    from .compute_spline_basis import compute_spline_basis
    from .spline_quadratic_gpu import spline_quadratic_gpu
    from .spline_cubic_gpu import spline_cubic_gpu


def spline(input, kernel_size, is_open_spline, K, degree, basis_kernel):
    if input.is_cuda:
        return compute_spline_basis(input, kernel_size, is_open_spline, K, basis_kernel)
    else:
        raise NotImplementedError()
