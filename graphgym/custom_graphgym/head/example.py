import torch.nn as nn

from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config

@register_head('ring_transfer')
class GNNRingHead(nn.Module):
    """
    GNN prediction head for ring transfer dataset task.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = nn.Linear(dim_in, dim_out)

    def forward(self, batch):
        x = batch.x[batch.mask]
        x = self.layer_post_mp(x)
        pred, label = x, batch.y
        return pred, label

###########################################

@register_head('head')
class ExampleNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = nn.Linear(dim_in, dim_out, bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.x[batch.node_label_index], batch.node_label
        else:
            return batch.x[batch.node_label_index], \
                batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
