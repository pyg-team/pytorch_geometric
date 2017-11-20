import torch

if torch.cuda.is_available():
    from .edgewise_spline_weighting_gpu import EdgewiseSplineWeightingGPU


def edgewise_spline_weighting(input, weight, amount, index):
    if input.is_cuda:
        return EdgewiseSplineWeightingGPU(amount, index)(input, weight)
    else:
        raise NotImplementedError
