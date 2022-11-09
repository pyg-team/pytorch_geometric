"""
This file is modified from HuggingFace `transformers/optimization.py`.
Copyright (c) the Google AI Language team authors, the HuggingFace Inc. team,
and the PyG Team. Licensed under the Apache License, Version 2.0.
"""
import functools
import inspect
import math
from typing import Any, Optional, Union

import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def lr_scheduler_resolver(
    query: Union[Any, str],
    optimizer: Optimizer,
    *args,
    warmup_ratio_or_steps: Optional[Union[float, int]] = 0.1,
    num_training_steps: Optional[int] = None,
    **kwargs,
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    r""" A helper to get any learning rate scheduler implemented in pyg and
    pytorch from its name or type.

    Args:
        query (Any or str): The query name of the learning rate scheduler.
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
    args = (optimizer, *args)
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

    base_cls = _LRScheduler
    base_cls_repr = 'LR'
    classes = [
        scheduler for scheduler in vars(torch.optim.lr_scheduler).values()
        if isinstance(scheduler, type) and issubclass(scheduler, base_cls)
    ]
    customized_lr_schedulers = [
        ConstantWithWarmupLR,
        LinearWithWarmupLR,
        CosineWithWarmupLR,
        CosineWithWarmupRestartsLR,
        PolynomialWithWarmupLR,
    ]
    classes += [ReduceLROnPlateau] + customized_lr_schedulers

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ''
    base_cls_repr = normalize_string(base_cls_repr)

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                if cls in customized_lr_schedulers:
                    cls_keys = inspect.signature(cls).parameters.keys()
                    if 'num_warmup_steps' in cls_keys:
                        kwargs['num_warmup_steps'] = num_warmup_steps
                    if 'num_training_steps' in cls_keys:
                        kwargs['num_training_steps'] = num_training_steps
                obj = cls(*args, **kwargs)
                return obj
            return cls

    choices = set(cls.__name__ for cls in classes)
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


class ConstantWithWarmupLR(LambdaLR):
    r""" Create a lr scheduler with a constant learning rate preceded by a
    warmup period during which the learning rate increases linearly between 0
    and the initial lr set in the optimizer.

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
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
