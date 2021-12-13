from torch_geometric.nn import (global_add_pool, global_mean_pool,
                                global_max_pool)
from torch_geometric.graphgym.register import register_pooling

register_pooling('add', global_add_pool)
register_pooling('mean', global_mean_pool)
register_pooling('max', global_max_pool)
