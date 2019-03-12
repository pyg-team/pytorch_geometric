import torch


class JanossyPooling(torch.nn.Module):
    def __init__(self, op, num_permutations, bias=True):
        super(JanossyPooling, self).__init__()

        self.op = op
        self.num_permutations = num_permutations

        self.reset_parameters()

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, x, edge_index, *args, **kwargs):
        """"""
        out = None
        for i in range(self.num_permutations):
            perm = torch.randperm(x.size(0), device=x.device)
            tmp = self.op(x[perm], perm[edge_index], *args, **kwargs)
            out = tmp if out is None else out + tmp
        return out / self.num_permutations

    def __repr__(self):
        return '{}({}, num_permutations={})'.format(
            self.__class__.__name__, self.op, self.num_permutations)
