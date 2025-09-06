import warnings

import torch

try:
    import lightning.pytorch as pl
    _pl_is_available = True
except ImportError:
    try:
        import pytorch_lightning as pl
        _pl_is_available = True
    except ImportError:
        _pl_is_available = False

if _pl_is_available:
    LightningModule = pl.LightningModule
    Callback = pl.Callback
else:
    pl = object
    LightningModule = torch.nn.Module
    Callback = object

    warnings.warn(
        "To use GraphGym, install 'pytorch_lightning' or 'lightning' via "
        "'pip install pytorch_lightning' or 'pip install lightning'",
        stacklevel=2)
