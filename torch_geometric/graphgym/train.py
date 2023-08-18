import warnings
from typing import Optional

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
    """
    LightningDataModule for handling data loading in GraphGym.

    This class provides data loaders for training, validation, and testing, which are
    created using the `create_loader` function. The data loaders can be accessed through
    the `train_dataloader`, `val_dataloader`, and `test_dataloader` methods, respectively.

    Note:
        Make sure to call the constructor of this class with appropriate configurations
        before using it.

    Example:
        >>> data_module = GraphGymDataModule()
        >>> model = GraphGymModule()
        >>> train(model, datamodule=data_module)

    Attributes:
        loaders: List of DataLoader instances for train, validation, and test.
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


def train(model: GraphGymModule, datamodule, logger: bool = True,
          trainer_config: Optional[dict] = None):
          """
    Train a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model to be trained.
        datamodule (GraphGymDataModule): The data module containing data loaders.
        logger (bool): Whether to enable logging during training.
        trainer_config (dict, optional): Additional trainer configurations.

    This function trains the provided GraphGym model using PyTorch Lightning. It sets up
    the trainer with given configurations, including callbacks for logging and checkpointing.
    After training, the function also tests the model using the provided data module.

    Example:
        >>> data_module = GraphGymDataModule()
        >>> model = GraphGymModule()
        >>> train(model, datamodule=data_module)

    Note:
        Make sure the appropriate configurations (`cfg`) are set before calling this
        function.
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
