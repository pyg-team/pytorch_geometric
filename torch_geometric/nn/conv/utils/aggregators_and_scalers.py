import torch
from torch_scatter import scatter_sum, scatter_mean, scatter_max, scatter_min

EPS = 1e-5


def aggregate_sum(src, index, dim, dim_size):
    return scatter_sum(src=src, index=index, dim=dim, out=None, dim_size=dim_size)


def aggregate_mean(src, index, dim, dim_size):
    return scatter_mean(src=src, index=index, dim=dim, out=None, dim_size=dim_size)


def aggregate_max(src, index, dim, dim_size):
    return scatter_max(src=src, index=index, dim=dim, out=None, dim_size=dim_size)[0]


def aggregate_min(src, index, dim, dim_size):
    return scatter_min(src=src, index=index, dim=dim, out=None, dim_size=dim_size)[0]


def aggregate_var(src, index, dim, dim_size):
    mean = aggregate_mean(src, index, dim, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim, dim_size)
    var = mean_squares - mean * mean
    return var


def aggregate_std(src, index, dim, dim_size):
    var = aggregate_var(src, index, dim, dim_size)
    out = torch.sqrt(torch.relu(var) + EPS)
    return out


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var}


def get_degree(src, index, dim, dim_size):
    # returns a tensor with the various degrees of the nodes
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)  # ensure no 0s
    count = count.unsqueeze(-1).unsqueeze(-1)
    return count


def scale_identity(src, D, avg_d=None):
    return src


def scale_amplification(src, D, avg_d=None):
    # log(D + 1) / d * X     where d is the average of the ``log(D + 1)`` in the training set
    scale = (torch.log(D + 1) / avg_d["log"])
    out = src * scale
    return out


def scale_attenuation(src, D, avg_d=None):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    scale = (avg_d["log"] / torch.log(D + 1))
    out = src * scale
    return out


def scale_linear(src, D, avg_d=None):
    scale = D / avg_d["lin"]
    out = src * scale
    return out


def scale_inverse_linear(src, D, avg_d=None):
    scale = avg_d["lin"] / D
    out = src * scale
    return out


SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation,
           'linear': scale_linear, 'inverse_linear': scale_inverse_linear}
