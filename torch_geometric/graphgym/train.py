import warnings
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule


class GraphGymDataModule(LightningDataModule):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading routines in GraphGym.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]


def train(
    model: GraphGymModule,
    datamodule: GraphGymDataModule,
    logger: bool = True,
    trainer_config: Optional[Dict[str, Any]] = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
        logger (bool, optional): Whether to enable logging during training.
            (default: :obj:`True`)
        trainer_config (dict, optional): Additional trainer configuration.
    """
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = []
    if logger:
        callbacks.append(LoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
        callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
