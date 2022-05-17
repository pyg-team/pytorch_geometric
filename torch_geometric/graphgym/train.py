from typing import Optional

from torch.utils.data import DataLoader

from torch_geometric.data.lightning_datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule


class GraphGymDataModule(LightningDataModule):
    def __init__(self):
        self._loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self._loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self._loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self._loaders[2]


def train(model: GraphGymModule, datamodule, logger: bool = True,
          training_config: Optional[dict] = None):
    logger_cbk = None
    if logger:
        logger_cbk = LoggerCallback()

    training_config = training_config or {}

    trainer = pl.Trainer(**training_config,
                         enable_checkpointing=cfg.train.enable_ckpt,
                         callbacks=logger_cbk, default_root_dir=cfg.out_dir,
                         max_epochs=cfg.optim.max_epoch)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
