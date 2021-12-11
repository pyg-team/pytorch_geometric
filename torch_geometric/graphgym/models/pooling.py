import torch_geometric.nn as nn
from torch_geometric.graphgym.register import register_pooling

register_pooling('add', nn.global_add_pool)
register_pooling('mean', nn.global_mean_pool)
register_pooling('max', nn.global_max_pool)
