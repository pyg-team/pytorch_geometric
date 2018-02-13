import torch

if torch.cuda.is_available():
    from .edgewise_spline_weighting_gpu import EdgewiseSplineWeightingGPU


def edgewise_spline_weighting(input, weight, amount, index, k_fw, k_bw):
    if input.is_cuda:
        K, M_in, M_out = weight.size()
        return EdgewiseSplineWeightingGPU(amount, index, K, M_in, M_out
                                          ,k_fw,k_bw)(input, weight)
    else:
        raise NotImplementedError
