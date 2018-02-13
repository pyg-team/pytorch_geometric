import torch

if torch.cuda.is_available():
    from .edgewise_spline_weighting_gpu import EdgewiseSplineWeightingGPU


def edgewise_spline_weighting(input, weight, amount, index):
    if input.is_cuda:
        K, M_in, M_out = weight.size()
        k_max = amount.size(1)
        return EdgewiseSplineWeightingGPU(amount, index, K, M_in, M_out,
                                          k_max)(input, weight)
    else:
        raise NotImplementedError
