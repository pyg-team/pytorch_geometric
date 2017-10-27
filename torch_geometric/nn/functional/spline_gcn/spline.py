import torch

from .spline_cpu import spline_cpu

if torch.cuda.is_available():
    from .spline_gpu import spline_gpu


def spline(input, kernel_size, is_open_spline, K, degree):
    if input.is_cuda:
        return spline_gpu(input, kernel_size, is_open_spline, K, degree)
    else:
        return spline_cpu(input, kernel_size, is_open_spline, degree)
