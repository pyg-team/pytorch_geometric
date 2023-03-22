import torch.nn as nn
import torch.nn.functional as F
from utils import IntegerQuantizer


class LinearQuantized(nn.Linear):
    """A quantizable linear layer"""
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True,
        qtype="INT8",
        signed=True,
        use_momentum=True,
        use_ste=True,
        symmetric=False,
        momentum=0.01,
        percentile=None,
        sample=None,
    ):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.inputs_quantizer = IntegerQuantizer(
            qtype,
            signed,
            use_momentum,
            use_ste,
            symmetric,
            momentum,
            percentile,
            sample,
        )

        self.weights_quantizer = IntegerQuantizer(
            qtype,
            signed,
            use_momentum,
            use_ste,
            symmetric,
            momentum,
            percentile,
            sample,
        )

        self.activation_quantizer = IntegerQuantizer(
            qtype,
            signed,
            use_momentum,
            use_ste,
            symmetric,
            momentum,
            percentile,
            sample,
        )

    def reset_parameters(self):
        super().reset_parameters()
        # TODO: RESET QUANTIZERS

    def forward(self, input):
        input_q = self.inputs_quantizer(input)
        w_q = self.weights_quantizer(self.weight)
        out = F.linear(input_q, w_q, self.bias)
        out = self.activation_quantizer(out)

        return out