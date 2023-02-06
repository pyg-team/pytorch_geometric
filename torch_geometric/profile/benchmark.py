import time
from typing import Any, Callable, List, Optional, Tuple

import torch


def benchmark(
    funcs: List[Callable],
    args: Tuple[Any],
    num_steps: int,
    func_names: Optional[List[str]] = None,
    num_warmups: int = 10,
    backward: bool = False,
):
    r"""Benchmark a list of functions :obj:`funcs` that receive the same set
    of arguments :obj:`args`.

    Args:
        funcs ([Callable]): The list of functions to benchmark.
        args ((Any, )): The arguments to pass to the functions.
        num_steps (int): The number of steps to run the benchmark.
        func_names ([str], optional): The names of the functions. If not given,
            will try to infer the name from the function itself.
            (default: :obj:`None`)
        num_warmups (int, optional): The number of warmup steps.
            (default: :obj:`10`)
        backward (bool, optional): If set to :obj:`True`, will benchmark both
            forward and backward passes. (default: :obj:`False`)
    """
    from tabulate import tabulate

    if num_steps <= 0:
        raise ValueError(f"'num_steps' must be a positive integer "
                         f"(got {num_steps})")

    if num_warmups <= 0:
        raise ValueError(f"'num_warmups' must be a positive integer "
                         f"(got {num_warmups})")

    if func_names is None:
        func_names = [func.__name__ for func in funcs]

    if len(funcs) != len(func_names):
        raise ValueError(f"Length of 'funcs' (got {len(funcs)}) and "
                         f"'func_names' (got {len(func_names)}) must be equal")

    ts: List[List[str]] = []
    for func, name in zip(funcs, func_names):
        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            args = [
                arg.detach().requires_grad_(backward)
                if arg.is_floating_point() else arg for arg in args
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = func(*args)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if backward:
                out_grad = torch.randn_like(out)
                t_start = time.perf_counter()

                out.backward(out_grad)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        ts.append([name, f'{t_forward:.4f}s'])
        if backward:
            ts[-1].append(f'{t_backward:.4f}s')

    header = ['Name', 'Forward']
    if backward:
        header.append('Backward')

    print(tabulate(ts, headers=header, tablefmt='psql'))
