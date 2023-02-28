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

        x = x.view(-1)
        probs = torch.tensor([prop], device=x.device).expand_as(x)
        out = torch.empty(probs.shape, dtype=torch.bool, device=probs.device)
        mask = torch.bernoulli(probs, out=out)
        return x[mask]

def get_qparams(max_val, min_val, num_bits, signed, eps, symmetric):
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)

    qmin = -(2.0 ** (num_bits - 1)) if signed else 0.0
    qmax = qmin + 2.0 ** num_bits - 1

    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:

        if symmetric:
            scale = 2 * max(abs(min_val), max_val) / (qmax - qmin)
            zero_point = 0.0 if signed else 128.0
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = max(scale, eps)
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
            zero_point = zero_point

    return qmin, qmax, zero_point, scale


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


quantize = Quantize.apply


SAMPLE_CUTOFF = 1000


def sample_tensor(prop, x):
    if x.numel() < SAMPLE_CUTOFF:
        return x

    cutoff_prop = SAMPLE_CUTOFF / x.numel()
    if cutoff_prop > prop:
        prop = cutoff_prop

    x = x.view(-1)
    probs = torch.tensor([prop], device=x.device).expand_as(x)
    out = torch.empty(probs.shape, dtype=torch.bool, device=probs.device)
    mask = torch.bernoulli(probs, out=out)
    return x[mask]


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

        x = x.view(-1)
        probs = torch.tensor([prop], device=x.device).expand_as(x)
        out = torch.empty(probs.shape, dtype=torch.bool, device=probs.device)
        mask = torch.bernoulli(probs, out=out)
        return x[mask]

    @staticmethod
    def get_qparams(max_val, min_val, num_bits, signed, eps, symmetric):
        min_val = min(0.0, float(min_val))
        max_val = max(0.0, float(max_val))

    qmin = -(2.0**(num_bits - 1)) if signed else 0.0
    qmax = qmin + 2.0**num_bits - 1

        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            if symmetric:
                scale = 2 * max(abs(min_val), max_val) / (qmax - qmin)
                zero_point = 0.0 if signed else 128.0
            else:
                scale = (max_val - min_val) / float(qmax - qmin)
                scale = max(scale, eps)
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)
                zero_point = zero_point

        return qmin, qmax, zero_point, scale
    
    def update_ranges(self, input):

        # updating min/max ranges
        min_val = self.min_val
        max_val = self.max_val

        input = self.sample_fn(input)
        current_min = self.min_fn(input)
        current_max = self.max_fn(input)

        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = current_min
            max_val = current_max
        else:
            if self.momentum_min_max:
                min_val = min_val + self.momentum * (current_min - min_val)
                max_val = max_val + self.momentum * (current_max - max_val)
            else:
                # Range update equivalent to PyTorch's MinMaxObserver
                # https://github.com/pytorch/pytorch/blob/9e5e5a7d9628f988a928969d09ff2bffe362c08c/torch/quantization/observer.py#L398
                min_val = torch.min(current_min, min_val)
                max_val = torch.max(current_max, max_val)

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        if self.training:
            self.update_ranges(input.detach())

        return Quantize.apply(
            input,
            self.max_val,
            self.min_val,
            self.num_bits,
            self.signed,
            self.eps,
            self.symmetric,
            self.ste,
        )

class Quantize(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, max_val, min_val, num_bits, signed, eps,
                symmetric, ste):
        output = input.clone()

        # compute qparams
        qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, num_bits,
                                                    signed, eps, symmetric)

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
        (input, ) = ctx.saved_tensors

        mask = input.clone()
        inv_scale = 1.0 / ctx.scale
        mask.mul_(inv_scale).add_(ctx.zp).round_()

        # gradient clipping
        grad_input = grad_output.clone()
        grad_input[mask.ge(ctx.qmax)] = 0
        grad_input[mask.le(ctx.qmin)] = 0

        return grad_input, None, None, None, None, None, None, None