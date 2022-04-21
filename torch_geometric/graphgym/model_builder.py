import warnings
from typing import Any, Tuple

import torch

from torch_geometric.graphgym import (
    compute_loss,
    create_optimizer,
    create_scheduler,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.register import network_dict, register_network

try:
    from pytorch_lightning import LightningModule
except ImportError:
    LightningModule = object
    warnings.warn("Please install 'pytorch_lightning' for using the GraphGym "
                  "experiment manager via 'pip install pytorch_lightning'")

register_network('gnn', GNN)


class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return optimizer, scheduler

    def _shared_step(self, batch, split: str):
        batch.split = split
        pred, true = self.forward(batch)
        loss, pred_score = compute_loss(pred, true)
        log_dict = dict(
            true=true,
            pred=pred_score,
            loss=loss,
        )
        return loss, log_dict

    def training_step(self, batch):
        loss, log_dict = self._shared_step(batch, "train")
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch):
        loss, log_dict = self._shared_step(batch, "val")
        self.log_dict(log_dict)
        return loss

    def test_step(self, batch):
        loss, log_dict = self._shared_step(batch, "test")
        self.log_dict(log_dict)
        return loss

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
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
