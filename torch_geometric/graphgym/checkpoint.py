from typing import List, Dict, Optional, Union, Any

import os
import glob
import os.path as osp

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

    if not osp.exists(get_ckpt_path(epoch)):
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

    os.makedirs(get_ckpt_dir(), exist_ok=True)
    torch.save(ckpt, get_ckpt_path(epoch))


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


def get_last_ckpt_epoch() -> int:
    return get_ckpt_epochs[-1]
