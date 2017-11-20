import torch

if torch.cuda.is_available():
    from .spline_linear_gpu import spline_linear_gpu
    from .spline_quadratic_gpu import spline_quadratic_gpu
    from .spline_cubic_gpu import spline_cubic_gpu


def spline(input, kernel_size, is_open_spline, K, degree):
    if input.is_cuda:
        if degree == 1:
            return spline_linear_gpu(input, kernel_size, is_open_spline, K)
        if degree == 2:
            return spline_quadratic_gpu(input, kernel_size, is_open_spline, K)
        if degree == 3:
            return spline_cubic_gpu(input, kernel_size, is_open_spline, K)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
