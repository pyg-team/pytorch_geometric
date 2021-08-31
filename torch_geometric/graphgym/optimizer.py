from torch_geometric.graphgym.config import cfg

import torch.optim as optim

import torch_geometric.graphgym.register as register


def create_optimizer(params):
    r"""
    Create optimizer for the model

    Args:
        params: PyTorch model parameters

    Returns: PyTorch optimizer

    """
    params = filter(lambda p: p.requires_grad, params)
    # Try to load customized optimizer
    for func in register.optimizer_dict.values():
        optimizer = func(params)
        if optimizer is not None:
            return optimizer
    if cfg.optim.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=cfg.optim.base_lr,
                               weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=cfg.optim.base_lr,
                              momentum=cfg.optim.momentum,
                              weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError('Optimizer {} not supported'.format(
            cfg.optim.optimizer))

    return optimizer


def create_scheduler(optimizer):
    r"""
    Create learning rate scheduler for the optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns: PyTorch scheduler

    """
    # Try to load customized scheduler
    for func in register.scheduler_dict.values():
        scheduler = func(optimizer)
        if scheduler is not None:
            return scheduler
    if cfg.optim.scheduler == 'none':
        scheduler = \
            optim.lr_scheduler.StepLR(optimizer,
                                      step_size=cfg.optim.max_epoch + 1)
    elif cfg.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.optim.steps,
                                                   gamma=cfg.optim.lr_decay)
    elif cfg.optim.scheduler == 'cos':
        scheduler = \
            optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=cfg.optim.max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.optim.scheduler))
    return scheduler
