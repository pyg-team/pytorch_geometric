import inspect
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn.lr_scheduler import (
    ConstantWithWarmupLR,
    CosineWithWarmupLR,
    CosineWithWarmupRestartsLR,
    LinearWithWarmupLR,
    PolynomialWithWarmupLR,
)
from torch_geometric.resolver import normalize_string, resolver

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.aggr as aggr
    if isinstance(query, (list, tuple)):
        return aggr.MultiAggregation(query, *args, **kwargs)

    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    aggr_dict = {
        'add': aggr.SumAggregation,
    }
    return resolver(aggrs, aggr_dict, query, base_cls, None, *args, **kwargs)


# Optimizer Resolver ##########################################################


def optimizer_resolver(query: Union[Any, str], *args, **kwargs):
    base_cls = Optimizer
    optimizers = [
        optimizer for optimizer in vars(torch.optim).values()
        if isinstance(optimizer, type) and issubclass(optimizer, base_cls)
    ]
    return resolver(optimizers, {}, query, base_cls, None, *args, **kwargs)


# Learning Rate Scheduler Resolver ############################################


def lr_scheduler_resolver(
    query: Union[Any, str],
    optimizer: Optimizer,
    warmup_ratio_or_steps: Optional[Union[float, int]] = 0.1,
    num_training_steps: Optional[int] = None,
    **kwargs,
) -> Union[LRScheduler, ReduceLROnPlateau]:
    r"""A resolver to obtain a learning rate scheduler implemented in either
    PyG or PyTorch from its name or type.

    Args:
        query (Any or str): The query name of the learning rate scheduler.
        optimizer (Optimizer): The optimizer to be scheduled.
        warmup_ratio_or_steps (float or int, optional): The number of warmup
            steps. If given as a `float`, it will act as a ratio that gets
            multiplied with the number of training steps to obtain the number
            of warmup steps. Only required for warmup-based LR schedulers.
            (default: :obj:`0.1`)
        num_training_steps (int, optional): The total number of training steps.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the LR scheduler.
    """
    if not isinstance(query, str):
        return query

    if isinstance(warmup_ratio_or_steps, float):
        if warmup_ratio_or_steps < 0 or warmup_ratio_or_steps > 1:
            raise ValueError(f"`warmup_ratio_or_steps` needs to be between "
                             f"0.0 and 1.0 when given as a floating point "
                             f"number (got {warmup_ratio_or_steps}).")
        if num_training_steps is not None:
            warmup_steps = round(warmup_ratio_or_steps * num_training_steps)
    elif isinstance(warmup_ratio_or_steps, int):
        if warmup_ratio_or_steps < 0:
            raise ValueError(f"`warmup_ratio_or_steps` needs to be positive "
                             f"when given as an integer "
                             f"(got {warmup_ratio_or_steps}).")
        warmup_steps = warmup_ratio_or_steps
    else:
        raise ValueError(f"Found invalid type of `warmup_ratio_or_steps` "
                         f"(got {type(warmup_ratio_or_steps)})")

    base_cls = LRScheduler
    classes = [
        scheduler for scheduler in vars(torch.optim.lr_scheduler).values()
        if isinstance(scheduler, type) and issubclass(scheduler, base_cls)
    ] + [ReduceLROnPlateau]

    customized_lr_schedulers = [
        ConstantWithWarmupLR,
        LinearWithWarmupLR,
        CosineWithWarmupLR,
        CosineWithWarmupRestartsLR,
        PolynomialWithWarmupLR,
    ]
    classes += customized_lr_schedulers

    query_repr = normalize_string(query)
    base_cls_repr = normalize_string('LR')

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                if cls in customized_lr_schedulers:
                    cls_keys = inspect.signature(cls).parameters.keys()
                    if 'num_warmup_steps' in cls_keys:
                        kwargs['num_warmup_steps'] = warmup_steps
                    if 'num_training_steps' in cls_keys:
                        kwargs['num_training_steps'] = num_training_steps
                obj = cls(optimizer, **kwargs)
                return obj
            return cls

    choices = {cls.__name__ for cls in classes}
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")
