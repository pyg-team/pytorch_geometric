from typing import Optional

import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

class IntegerQuantizer(nn.Module):
    """Allows for per-tensor integer uniform (symmetric or asymmetric/affine) quantization."""
    def __init__(
        self,
        qtype: str,
        signed: bool,
        use_momentum: bool,
        use_ste: bool,
        symmetric: bool = False,
        momentum: float = 0.01,
        percentile: Optional[float] = None,
        sample: Optional[float] = None,
    ):
        super(IntegerQuantizer, self).__init__()

        qtype_to_num_bits = {
            'INT4': 4,
            'INT8': 8,
        }
        assert(qtype in qtype_to_num_bits, f"Invalid qtype: {qtype}")

        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.momentum = momentum
        self.num_bits = num_bits
        self.signed = signed
        self.symmetric = symmetric
        self.eps = torch.finfo(torch.float32).eps

        self.ste = use_ste
        self.momentum_min_max = use_momentum

        if percentile is None:
            self.min_fn = torch.min
            self.max_fn = torch.max
        else:
            self.min_fn = lambda t: torch.kthvalue(
                torch.flatten(t), max(1, min(t.numel(), int(t.numel() * percentile)))
            )[0]
            self.max_fn = lambda t: torch.kthvalue(
                torch.flatten(t),
                min(t.numel(), max(1, int(t.numel() * (1 - percentile)))),
            )[0]

        if sample is None:
            self.sample_fn = lambda x: x
        else:
            assert percentile is not None
            self.sample_fn = lambda x: sample_tensor(sample, x)

    @staticmethod
    def sample_tensor(prop, x, sample_cutoff=1000):
        if x.numel() < sample_cutoff:
            return x

        cutoff_prop = sample_cutoff / x.numel()
        if cutoff_prop > prop:
            prop = cutoff_prop

class Quantize(InplaceFunction):
    @classmethod
    def forward(
        cls, ctx, input, max_val, min_val, num_bits, signed, eps, symmetric, ste
    ):
        output = input.clone()

        # compute qparams
        qmin, qmax, zero_point, scale = get_qparams(
            max_val, min_val, num_bits, signed, eps, symmetric
        )

        # save stuff for backprop (if STE not enabled)
        ctx.STE = ste
        if not ste:
            ctx.save_for_backward(input)
            ctx.qmin = qmin
            ctx.qmax = qmax
            ctx.scale = scale
            ctx.zp = zero_point

        inv_scale = 1.0 / scale

        output.mul_(inv_scale).add_(zero_point)
        output.round_().clamp_(qmin, qmax)  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.STE:
            return grad_output, None, None, None, None, None, None, None

        # Applying gradient clippling as described here:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cuda/fake_quantize_core.cu
        (input,) = ctx.saved_tensors

        mask = input.clone()
        inv_scale = 1.0 / ctx.scale
        mask.mul_(inv_scale).add_(ctx.zp).round_()

        # gradient clipping
        grad_input = grad_output.clone()
        grad_input[mask.ge(ctx.qmax)] = 0
        grad_input[mask.le(ctx.qmin)] = 0

        return grad_input, None, None, None, None, None, None, None