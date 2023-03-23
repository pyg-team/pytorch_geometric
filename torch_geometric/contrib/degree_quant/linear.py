from torch import nn 



class LinearQuantized(nn.Linear):
    """
    A quantizable linear layer
    
   
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    
    
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """
    
    
    
    
    
    def __init__(
        self, in_features, out_features, layer_quantizers, bias=True
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

        """
        Args: 
        input (torch.Tensor): The input node features.
        
        """

        input_q = self.layer_quant["inputs"](input)
        w_q = self.layer_quant["weights"](self.weight)
        out = F.linear(input_q, w_q, self.bias)
        out = self.layer_quant["features"](out)

        return out