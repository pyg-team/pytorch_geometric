import torch

from .edgewise_spline_weighting_cpu import EdgewiseSplineWeightingCPU

if torch.cuda.is_available():
    from .edgewise_spline_weighting_gpu import EdgewiseSplineWeightingGPU


def edgewise_spline_weighting(input, weight, amount, index):
    if input.is_cuda:
        return EdgewiseSplineWeightingGPU(amount, index)(input, weight)
    else:
        return EdgewiseSplineWeightingCPU(amount, index)(input, weight)
