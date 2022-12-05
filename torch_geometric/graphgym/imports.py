import warnings

import torch

try:
    import pytorch_lightning as pl
    LightningModule = pl.LightningModule
    LightningDataModule = pl.LightningDataModule
    Callback = pl.Callback
except ImportError:
    pl = object
    LightningModule = torch.nn.Module
    LightningDataModule = object
    Callback = object

    warnings.warn("Please install 'pytorch_lightning' for using the GraphGym "
                  "experiment manager via 'pip install pytorch_lightning'")
