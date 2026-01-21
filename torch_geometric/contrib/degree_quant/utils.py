import torch_scatter


def scatter_(name, src, index, dim=0, dim_size=None):
    """
    Method implemented to mask the minimum and maximum value in the tensor.
    Allows to use different scatter e
    """

    assert name in ["add", "mean", "min", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out
    if name == "max":
        out[out < -10000] = 0
    elif name == "min":
        out[out > 10000] = 0
    return out
