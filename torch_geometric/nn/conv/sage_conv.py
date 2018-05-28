import torch


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SAGEConv, self).__init__()
