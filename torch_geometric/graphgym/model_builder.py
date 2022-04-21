import logging

import torch
from torch import nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.register import network_dict, register_network

logger = logging.getLogger(__name__)

try:
    from pytorch_lightning import LightningModule
except ImportError:
    logger.warning(
        "To use GraphGym please pip install pytorch_lightning first")
    LightningModule = object

register_network('gnn', GNN)


class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> nn.Module:
        return self.model.pre_mp


def create_model(to_device=True, dim_in=None, dim_out=None):
    r"""
    Create model for graph machine learning

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.device))
    return model
