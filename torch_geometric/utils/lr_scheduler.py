"""
This file is modified from HuggingFace `transformers/optimization.py`.
Copyright (c) the Google AI Language team authors, the HuggingFace Inc. team,
and the PyG Team. Licensed under the Apache License, Version 2.0.
"""
import functools
import math
from enum import Enum
from typing import Optional, Union

from class_resolver.contrib.torch import lr_scheduler_resolver
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler


class LRSchedulerType(Enum):
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    LINEAR_WITH_WARMUP = "linear_with_warmup"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    COSINE_WITH_WARMUP_RESTARTS = "cosine_with_warmup_restarts"
    POLYNOMIAL_WITH_WARMUP = "polynomial_with_warmup"
    MAYBE_TORCH_LR_SCHEDULER = "maybe_torch_lr_scheduler"

    @classmethod
    def _missing_(cls, value):
        return cls.MAYBE_TORCH_LR_SCHEDULER


def _constant_lr_lambda_with_warmup(
    current_step: int,
    num_warmup_steps: int,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    r""" Create a lr scheduler with a constant learning rate preceded by a
    warmup period during which the learning rate increases linearly between 0
    and the initial lr set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    lr_lambda = functools.partial(
        _constant_lr_lambda_with_warmup,
        num_warmup_steps=num_warmup_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _linear_lr_lambda_with_warmup(
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


def get_linear_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    r""" Create a lr scheduler with a learning rate that decreases linearly
    from the initial lr set in the optimizer to 0, after a warmup period during
    which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    lr_lambda = functools.partial(
        _linear_lr_lambda_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _cosine_lr_lambda_with_warmup(
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
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def get_cosine_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    r""" Create a lr scheduler with a learning rate that decreases following
    the values of the cosine function between the initial lr set in the
    optimizer to 0, after a warmup period during which it increases linearly
    between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float, optional): The number of waves in the cosine
            schedule. (the defaults is to decrease from the max value to 0
            following a half-cosine). (default: :obj:`0.5`)
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    lr_lambda = functools.partial(
        _cosine_lr_lambda_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _cosine_lr_lambda_with_hard_restarts_and_warmup(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
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


def get_cosine_lr_scheduler_with_hard_restarts_and_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 3,
    last_epoch: int = -1,
) -> LambdaLR:
    r""" Create a lr scheduler with a learning rate that decreases following
    the values of the cosine function between the initial lr set in the
    optimizer to 0, with several hard restarts, after a warmup period during
    which it increases linearly between 0 and the initial lr set in the
    optimizer.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (int, optional): The number of hard restarts to use.
            (default: :obj:`3`)
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """
    lr_lambda = functools.partial(
        _cosine_lr_lambda_with_hard_restarts_and_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _polynomial_lr_lambda_with_warmup(
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
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    r""" Create a lr scheduler with a learning rate that decreases as a
    polynomial decay from the initial lr set in the optimizer to end lr defined
    by `lr_end`, after a warmup period during which it increases linearly from
    0 to the initial lr set in the optimizer.

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

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(
            f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    lr_lambda = functools.partial(
        _polynomial_lr_lambda_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_init=lr_init,
        lr_end=lr_end,
        power=power,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_LR_SCHEDULER_HELPER = {
    LRSchedulerType.CONSTANT_WITH_WARMUP:
    get_constant_lr_scheduler_with_warmup,
    LRSchedulerType.LINEAR_WITH_WARMUP: get_linear_lr_scheduler_with_warmup,
    LRSchedulerType.COSINE_WITH_WARMUP: get_cosine_lr_scheduler_with_warmup,
    LRSchedulerType.COSINE_WITH_WARMUP_RESTARTS:
    get_cosine_lr_scheduler_with_hard_restarts_and_warmup,
    LRSchedulerType.POLYNOMIAL_WITH_WARMUP:
    get_polynomial_lr_scheduler_with_warmup,
    LRSchedulerType.MAYBE_TORCH_LR_SCHEDULER: lr_scheduler_resolver,
}


def get_lr_scheduler(
    scheduler_name: Union[str, LRSchedulerType],
    optimizer: Optimizer,
    warmup_ratio_or_steps: Optional[Union[float, int]] = 0.1,
    num_training_steps: Optional[int] = None,
    **kwargs,
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    r""" A helper to get any learning rate scheduler implemented in pyg and
    pytorch from its name or type.

    Args:
        scheduler_name (str or LRSchedulerType): The name or type of the
            learning rate scheduler.
        optimizer (Optimizer): The optimizer to be scheduled.
        warmup_ratio_or_steps (float or int, optional): The hyperparameter to
            determine the `num_warmup_steps`. If a `float` type is set, it will
            act as a ratio to be multiplied with the `num_training_steps` to
            get the `num_warmup_steps`. If a `int` type is set, it will act as
            the `num_warmup_steps`. It is only required for warmup-based lr
            schedulers. (default: :obj:`0.1`)
        num_training_steps (int, optional): The total `num_training_steps` to
            used for the lr scheduler. This is not required by all
            schedulers. (default: :obj:`None`)
         **kwargs (optional): Additional arguments of the lr scheduler.
    """
    scheduler_type = LRSchedulerType(scheduler_name)
    scheduler_helper = TYPE_TO_LR_SCHEDULER_HELPER[scheduler_type]

    # Not a customized lr scheduler. Will try to resolve from `torch.optim`.
    if scheduler_type == LRSchedulerType.MAYBE_TORCH_LR_SCHEDULER:
        return scheduler_helper.make(
            scheduler_name,
            optimizer=optimizer,
            **kwargs,
        )

    # All other schedulers require `warmup_ratio_or_steps`.
    if warmup_ratio_or_steps is None:
        raise ValueError(f"{scheduler_name} requires `warmup_ratio_or_steps`.")

    if isinstance(warmup_ratio_or_steps, float):
        if warmup_ratio_or_steps < 0.0 or warmup_ratio_or_steps > 1.0:
            raise ValueError(
                f"Got `warmup_ratio_or_steps`={warmup_ratio_or_steps}."
                "`warmup_ratio_or_steps` is using as ratio which should be "
                "a floating number between 0.0 and 1.0.")
        num_warmup_steps = round(warmup_ratio_or_steps * num_training_steps)
    if isinstance(warmup_ratio_or_steps, int):
        if warmup_ratio_or_steps < 0:
            raise ValueError(
                f"Got `warmup_ratio_or_steps`={warmup_ratio_or_steps}."
                "`warmup_ratio_or_steps` is using as the num of steps which "
                "should be an integer number larger than 0.")
        num_warmup_steps = warmup_ratio_or_steps

    if scheduler_type == LRSchedulerType.CONSTANT_WITH_WARMUP:
        return scheduler_helper(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`.
    if num_training_steps is None:
        raise ValueError(f"{scheduler_name} requires `num_training_steps`.")

    return scheduler_helper(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
