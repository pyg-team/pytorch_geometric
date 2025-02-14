import warnings

import torch

try:
    import lighting as L
    LightningModule = L.LightningModule
    Callback = L.Callback
except ImportError:
    L = object
    LightningModule = torch.nn.Module
    Callback = object

    warnings.warn("Please install 'lightning' via  "
                  "'pip install lightning' in order to use GraphGym")
