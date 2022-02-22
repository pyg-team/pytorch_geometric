from torch_geometric.graphgym.config import cfg


def is_train_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the training epoch."""
    return is_eval_epoch(cur_epoch) or not cfg.train.skip_train_eval


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return ((cur_epoch + 1) % cfg.train.eval_period == 0 or cur_epoch == 0
            or (cur_epoch + 1) == cfg.optim.max_epoch)


def is_ckpt_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return ((cur_epoch + 1) % cfg.train.ckpt_period == 0
            or (cur_epoch + 1) == cfg.optim.max_epoch)
