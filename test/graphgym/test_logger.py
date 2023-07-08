from torch_geometric.graphgym.logger import Logger, LoggerCallback
from torch_geometric.testing import withPackage


@withPackage('yacs', 'pytorch_lightning')
def test_logger_callback():
    logger = LoggerCallback()
    assert isinstance(logger.train_logger, Logger)
    assert isinstance(logger.val_logger, Logger)
    assert isinstance(logger.test_logger, Logger)
