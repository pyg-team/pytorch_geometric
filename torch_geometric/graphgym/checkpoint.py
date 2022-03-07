<<<<<<< HEAD
from typing import List, Dict, Optional, Union, Any

<<<<<<< HEAD
import torch
# import os
from pathlib import Path
import logging


def get_ckpt_dir():
    return '{}/ckpt'.format(cfg.run_dir)


def get_all_epoch():
    d = get_ckpt_dir()
    names = Path.iterdir(Path(d)) if Path(d).exists() else []
    if len(names) == 0:
        return [0]
    epochs = [int(name.split('.')[0]) for name in names]
    return epochs


def get_last_epoch():
    return max(get_all_epoch())
=======
import os
=======
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd
import glob
import os
import os.path as osp
<<<<<<< HEAD
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec
=======
from typing import Any, Dict, List, Optional, Union
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd

import torch

from torch_geometric.graphgym.config import cfg

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
SCHEDULER_STATE = 'scheduler_state'


def load_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = -1,
) -> int:
    r"""Loads the model checkpoint at a given epoch."""
    epoch = get_ckpt_epoch(epoch)
    path = get_ckpt_path(epoch)

<<<<<<< HEAD
<<<<<<< HEAD
    '''
    if cfg.train.epoch_resume < 0:
        epoch_resume = get_last_epoch()
    else:
        epoch_resume = cfg.train.epoch_resume
    ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch_resume)
    if not Path(ckpt_name).is_file():
=======
    if not osp.exists(get_ckpt_path(epoch)):
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec
=======
    if not osp.exists(path):
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd
        return 0

    ckpt = torch.load(path)
    model.load_state_dict(ckpt[MODEL_STATE])
    if optimizer is not None and OPTIMIZER_STATE in ckpt:
        optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
    if scheduler is not None and SCHEDULER_STATE in ckpt:
        scheduler.load_state_dict(ckpt[SCHEDULER_STATE])

    return epoch + 1


def save_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
):
    r"""Saves the model checkpoint at a given epoch."""
    ckpt: Dict[str, Any] = {}
    ckpt[MODEL_STATE] = model.state_dict()
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()

<<<<<<< HEAD
    '''
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    Path.mkdirs(get_ckpt_dir(), exist_ok=True)
    ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
    torch.save(ckpt, ckpt_name)
    logging.info('Check point saved: {}'.format(ckpt_name))
=======
    os.makedirs(get_ckpt_dir(), exist_ok=True)
<<<<<<< HEAD
    torch.save(ckpt, get_ckpt_path(epoch))
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec
=======
    torch.save(ckpt, get_ckpt_path(get_ckpt_epoch(epoch)))


def remove_ckpt(epoch: int = -1):
    r"""Removes the model checkpoint at a given epoch."""
    os.remove(get_ckpt_path(get_ckpt_epoch(epoch)))
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd


def clean_ckpt():
    r"""Removes all but the last model checkpoint."""
    for epoch in get_ckpt_epochs()[:-1]:
        os.remove(get_ckpt_path(epoch))


###############################################################################


def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(epoch: Union[int, str]) -> str:
    return osp.join(get_ckpt_dir(), f'{epoch}.ckpt')


def get_ckpt_epochs() -> List[int]:
    paths = glob.glob(get_ckpt_path('*'))
    return sorted([int(osp.basename(path).split('.')[0]) for path in paths])


<<<<<<< HEAD
<<<<<<< HEAD
    '''
    epochs = get_all_epoch()
    epoch_last = max(epochs)
    for epoch in epochs:
        if epoch != epoch_last:
            ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
            Path(ckpt_name).unlink()
=======
def get_last_ckpt_epoch() -> int:
    return get_ckpt_epochs[-1]
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec
=======
def get_ckpt_epoch(epoch: int) -> int:
    if epoch < 0:
        epochs = get_ckpt_epochs()
        epoch = epochs[epoch] if len(epochs) > 0 else 0
    return epoch
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd
