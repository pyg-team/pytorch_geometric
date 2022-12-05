import warnings

try:
    import pytorch_lightning as pl
except ImportError:
    pl = object

    warnings.warn("Please install 'pytorch_lightning' for using the GraphGym "
                  "experiment manager via 'pip install pytorch_lightning'")
