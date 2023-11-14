from torch_geometric.graphgym.register import register_pooling
from torch_geometric.utils import scatter


@register_pooling('example')
def global_example_pool(x, batch, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='sum')
