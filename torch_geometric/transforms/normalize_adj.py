from torch_sparse import SparseTensor


def normalize_adj(adj: SparseTensor, symmetric: bool = True) -> SparseTensor:
    if symmetric:
        degree = adj.sum(dim=1).pow_(-0.5)
        degree.masked_fill_(degree == float('inf'), 0)
        return degree.view(-1, 1) * adj * degree.view(1, -1)
    else:
        degree = adj.sum(dim=1).pow_(-1)
        degree.masked_fill_(degree == float('inf'), 0)
        return degree.view(-1, 1) * adj


class NormalizeAdj(object):
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric

    def __call__(self, data):
        data.adj = normalize_adj(data.adj, self.symmetric)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(symmetric={self.symmetric})'
