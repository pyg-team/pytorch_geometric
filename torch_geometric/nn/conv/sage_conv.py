import torch


class SAGEMaxPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(x, edge_index):
        pass


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SAGEConv, self).__init__()

        # w needs to be 2 * in_channels x out_channels
        # bias out_channels

    def forward(self, x, edge_index):
        out = torch.cat([x, self.aggregate(x, edge_index)], dim=-1)
        out = torch.matmul(out, self.weight)
        out = out / torch.norm(out, p=2, dim=-1)

        if self.bias is not None:
            out = out + self.bias  # Do it for the others, too

        return x
