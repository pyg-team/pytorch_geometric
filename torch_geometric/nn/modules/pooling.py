import torch.nn.functional as F
from torch.nn import Module


class GraclusMaxPool(Module):
    def __init__(self, level):
        super(GraclusMaxPool, self).__init__()
        self.level = level
        self.kernel_size = 2**level

    def forward(self, input):
        output = F.max_pool1d(input.t().unsqueeze(0), self.kernel_size)
        return output.squeeze(0).t()

    def __repr__(self):
        s = '{name}(level={level})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
