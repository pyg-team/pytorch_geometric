# See HuggingFace `transformers/optimization.py`.
import functools
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantWithWarmupLR(LambdaLR):
    r"""Creates a LR scheduler with a constant learning rate preceded by a
    warmup period during which the learning rate increases linearly between
    :obj:`0` and the initial LR set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
    ) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0


class LinearWithWarmupLR(LambdaLR):
    r"""Creates a LR scheduler with a learning rate that decreases linearly
    from the initial LR set in the optimizer to :obj:`0`, after a warmup period
    during which it increases linearly from :obj:`0` to the initial LR set in
    the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)),
        )


class CosineWithWarmupLR(LambdaLR):
    r"""Creates a LR scheduler with a learning rate that decreases following
    the values of the cosine function between the initial LR set in the
    optimizer to :obj:`0`, after a warmup period during which it increases
    linearly between :obj:`0` and the initial LR set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float, optional): The number of waves in the cosine
            schedule (the default decreases LR from the max value to :obj:`0`
            following a half-cosine). (default: :obj:`0.5`)
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0,
            0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )


class CosineWithWarmupRestartsLR(LambdaLR):
    r"""Creates a LR scheduler with a learning rate that decreases following
    the values of the cosine function between the initial LR set in the
    optimizer to :obj:`0`, with several hard restarts, after a warmup period
    during which it increases linearly between :obj:`0` and the initial LR set
    in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (int, optional): The number of hard restarts to use.
            (default: :obj:`3`)
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 3,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int,
    ) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi *
                                  ((float(num_cycles) * progress) % 1.0))),
        )


class PolynomialWithWarmupLR(LambdaLR):
    r"""Creates a LR scheduler with a learning rate that decreases as a
    polynomial decay from the initial LR set in the optimizer to end LR defined
    by `lr_end`, after a warmup period during which it increases linearly from
    :obj:`0` to the initial LR set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        lr_end (float, optional): The end learning rate. (default: :obj:`1e-7`)
        power (float, optional): The power factor of the polynomial decay.
            (default: :obj:`1.0`)
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        lr_init = optimizer.defaults["lr"]
        if not (lr_init > lr_end):
            raise ValueError(f"`lr_end` ({lr_end}) must be smaller than the "
                             f"initial lr ({lr_init})")

        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_init=lr_init,
            lr_end=lr_end,
            power=power,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_init: float,
        lr_end: float,
        power: float,
    ) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # As `LambdaLR` multiplies by `lr_init`.
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # As `LambdaLR` multiplies by `lr_init`.
