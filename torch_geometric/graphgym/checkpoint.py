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
import glob
import os.path as osp
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec

import torch

from torch_geometric.graphgym.config import cfg


def load_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = -1,
) -> int:
    r"""Loads the model checkpoint at a given epoch."""
    if epoch < 0:
        epochs = get_ckpt_epochs()
        epoch = epochs[epoch] if len(epochs) > 0 else 0

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
        return 0

    ckpt = torch.load(get_ckpt_path(epoch))
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    return epoch + 1


def save_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
):
    r"""Saves the model checkpoint at a given epoch."""
    ckpt: Dict[str, Any] = {}
    ckpt['model_state'] = model.state_dict()
    if optimizer is not None:
        ckpt['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state'] = scheduler.state_dict()

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
    torch.save(ckpt, get_ckpt_path(epoch))
>>>>>>> 09ae5768f86a5f57a353ade1637d392d95cf9dec


def clean_ckpt():
    r"""Removes all but the last model checkpoint."""
    for epoch in get_ckpt_epochs()[:-1]:
        os.remove(get_ckpt_path(epoch))


###############################################################################


def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(name: Union[int, str]) -> str:
    return osp.join(get_ckpt_dir(), f'{name}.ckpt')


def get_ckpt_epochs() -> List[int]:
    paths = glob.glob(get_ckpt_path('*'))
    return sorted([int(osp.basename(path).split('.')[0]) for path in paths])


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
