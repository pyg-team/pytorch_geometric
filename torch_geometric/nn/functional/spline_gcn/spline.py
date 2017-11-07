import torch

from .spline_cpu import spline_cpu

if torch.cuda.is_available():
    from .spline_gpu import spline_gpu
    from .spline_quadratic_gpu import spline_quadratic_gpu


def spline(input, kernel_size, is_open_spline, K, degree):
    if input.is_cuda:
        if degree == 1:
            return spline_gpu(input, kernel_size, is_open_spline, K)
        if degree == 2:
            return spline_quadratic_gpu(input, kernel_size, is_open_spline, K)
        else:
            raise NotImplementedError()
    else:
        return spline_cpu(input, kernel_size, is_open_spline, degree)
