import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('smoothl1')
def loss_example(pred, true):
    if cfg.model.loss_fun == 'smoothl1':
        l1_loss = torch.nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
