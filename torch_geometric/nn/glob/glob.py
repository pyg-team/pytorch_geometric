from torch_geometric.utils import scatter_


def global_add_pool(x, batch, size=None):
    size = batch[-1].item() + 1 if size is None else size
    return scatter_('add', x, batch, dim_size=size)


def global_mean_pool(x, batch, size=None):
    size = batch[-1].item() + 1 if size is None else size
    return scatter_('mean', x, batch, dim_size=size)


def global_max_pool(x, batch, size=None):
    size = batch[-1].item() + 1 if size is None else size
    return scatter_('max', x, batch, dim_size=size)
