import torch.optim as optim

from graphgym.register import register_optimizer, register_scheduler

from graphgym.config import cfg


def optimizer_example(params):
    if cfg.optim.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=cfg.optim.base_lr,
                                  weight_decay=cfg.optim.weight_decay)
        return optimizer


register_optimizer('adagrad', optimizer_example)


def scheduler_example(optimizer):
    if cfg.optim.optimizer == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return scheduler


register_scheduler('reduce', scheduler_example)
