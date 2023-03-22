
import  torch 
from torch.autograd.function import InplaceFunction
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch_geometric.utils import degree
from utils import get_qparams, sample_tensor
from torch_geometric.data import Batch
from torch.nn import Identity


# This class is used to update the gradient using the quanitzation approach.
class Quantize(InplaceFunction):

    """
    A Qauntization Aware Training Class for Computing the Forward and Backward Gradient Steps
    on the Quantized Integer Tensor.
    
    """
    @classmethod
    def forward(
        cls, ctx, input, max_val, min_val, num_bits, signed, eps, symmetric, ste=False
    ):
        output = input.clone()

        qmin, qmax, zero_point, scale = Quantize.get_qparams(
            max_val, min_val, num_bits, signed, eps, symmetric
        )

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

        (input,) = ctx.saved_tensors

        mask = input.clone()
        inv_scale = 1.0 / ctx.scale
        mask.mul_(inv_scale).add_(ctx.zp).round_()

        # gradient clipping
        grad_input = grad_output.clone()
        grad_input[mask.ge(ctx.qmax)] = 0
        grad_input[mask.le(ctx.qmin)] = 0

        return grad_input, None, None, None, None, None, None, None

    @staticmethod
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





# The main Quantization Module which is used to change precision to Integer (8-bits or 4-bits)
class IntegerQuantizer(nn.Module):
    """
    This Class is used to Quantize an input tensor which supports Quantization aware Training. 
    Using this, the tensor can be quantized to INT8 precision using the default settings. 
    """

    def __init__(
        self,
        num_bits: int = 8,
        signed: bool = True,
        use_momentum: bool= True,
        use_ste: bool = False,
        symmetric: bool = False,
        momentum: float = 0.01,
        percentile: Optional[float] = None,
        sample: Optional[float] = None,
    ):
        super(IntegerQuantizer, self).__init__()
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
            self.sample_fn = lambda x: IntegerQuantizer.sample_tensor(sample, x)

    
    @staticmethod
    def sample_tensor(prop, x):
        if x.numel() < 1000:
            return x

        cutoff_prop = 1000 / x.numel()
        if cutoff_prop > prop:
            prop = cutoff_prop

        x = x.view(-1)
        probs = torch.tensor([prop], device=x.device).expand_as(x)
        out = torch.empty(probs.shape, dtype=torch.bool, device=probs.device)
        mask = torch.bernoulli(probs, out=out)
        return x[mask]
    
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
    
class LinearQuantized(nn.Linear):
    """A quantizable linear layer which uses the Integer Quantized Layer as the Input"""
    def __init__(
        self, in_features, out_features, layer_quantizers = IntegerQuantizer, bias=True
    ):
        # create quantization modules for this layer
        self.layer_quant_fns = layer_quantizers
        super(LinearQuantized, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.layer_quant = nn.ModuleDict()
        for key in ["inputs", "features", "weights"]:
            self.layer_quant[key] = self.layer_quant_fns[key]()

    def forward(self, input):
        input_q = self.layer_quant["inputs"](input)
        w_q = self.layer_quant["weights"](self.weight)
        out = F.linear(input_q, w_q, self.bias)
        out = self.layer_quant["features"](out)

        return out
    
# Generates the Mask based on the node degree. 
class ProbabilisticHighDegreeMask:
    def __init__(self, low_quantise_prob, high_quantise_prob, per_graph=True):
        self.low_prob = low_quantise_prob
        self.high_prob = high_quantise_prob
        self.per_graph = per_graph

    def _process_graph(self, graph):
        n = graph.num_nodes
        indegree = degree(graph.edge_index[1], n, dtype=torch.long)
        counts = torch.bincount(indegree)

        step_size = (self.high_prob - self.low_prob) / n
        indegree_ps = counts * step_size
        indegree_ps = torch.cumsum(indegree_ps, dim=0)
        indegree_ps += self.low_prob
        graph.prob_mask = indegree_ps[indegree]

        return graph

    def __call__(self, data):
        if self.per_graph and isinstance(data, Batch):
            graphs = data.to_data_list()
            processed = []
            for g in graphs:
                g = self._process_graph(g)
                processed.append(g)
            return Batch.from_data_list(processed)
        else:
            return self._process_graph(data)
        


def create_quantizer(qypte, ste, momentum, percentile, signed, sample_prop):
    if qypte == "FP32":
        return Identity
    else:
        return lambda: IntegerQuantizer(4 if qypte == "INT4" else 8, signed=signed,
                                        use_ste=ste,
                                        use_momentum=momentum,
                                        percentile=percentile,
                                        sample=sample_prop,
                                        )



def make_quantizers(qypte, dq, sign_input, ste, momentum, percentile, sample_prop):
    if dq:
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "inputs_low": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "norm_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "inputs_high": create_quantizer(
                "FP32", ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
            "features_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            
            "message_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "message_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
            "update_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "update_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
            "aggregate_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "aggregate_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
        }
    else:
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            "message": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "update_q": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "aggregate": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
    return layer_quantizers, mp_quantizers




keys = ['inputs', 'weight' , 'features', 'norm',
       'attention', 'message', 'aggregate', 'update']
options = ['low', 'high', 'q']

