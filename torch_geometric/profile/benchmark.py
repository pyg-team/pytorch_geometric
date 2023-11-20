import time
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import is_torch_sparse_tensor


def require_grad(x: Any, requires_grad: bool = True) -> Any:
    if (isinstance(x, Tensor) and x.is_floating_point()
            and not is_torch_sparse_tensor(x)):
        return x.detach().requires_grad_(requires_grad)
    elif isinstance(x, list):
        return [require_grad(v, requires_grad) for v in x]
    elif isinstance(x, tuple):
        return tuple(require_grad(v, requires_grad) for v in x)
    elif isinstance(x, dict):
        return {k: require_grad(v, requires_grad) for k, v in x.items()}
    return x


def benchmark(
    funcs: List[Callable],
    args: Union[Tuple[Any], List[Tuple[Any]]],
    num_steps: int,
    func_names: Optional[List[str]] = None,
    num_warmups: int = 10,
    backward: bool = False,
    per_step: bool = False,
    progress_bar: bool = False,
):
    r"""Benchmark a list of functions :obj:`funcs` that receive the same set
    of arguments :obj:`args`.

    Args:
        funcs ([Callable]): The list of functions to benchmark.
        args ((Any, ) or [(Any, )]): The arguments to pass to the functions.
            Can be a list of arguments for each function in :obj:`funcs` in
            case their headers differ.
            Alternatively, you can pass in functions that generate arguments
            on-the-fly (e.g., useful for benchmarking models on various sizes).
        num_steps (int): The number of steps to run the benchmark.
        func_names ([str], optional): The names of the functions. If not given,
            will try to infer the name from the function itself.
            (default: :obj:`None`)
        num_warmups (int, optional): The number of warmup steps.
            (default: :obj:`10`)
        backward (bool, optional): If set to :obj:`True`, will benchmark both
            forward and backward passes. (default: :obj:`False`)
        per_step (bool, optional): If set to :obj:`True`, will report runtimes
            per step. (default: :obj:`False`)
        progress_bar (bool, optional): If set to :obj:`True`, will print a
            progress bar during benchmarking. (default: :obj:`False`)
    """
    from tabulate import tabulate

    if num_steps <= 0:
        raise ValueError(f"'num_steps' must be a positive integer "
                         f"(got {num_steps})")

    if num_warmups <= 0:
        raise ValueError(f"'num_warmups' must be a positive integer "
                         f"(got {num_warmups})")

    if func_names is None:
        func_names = [get_func_name(func) for func in funcs]

    if len(funcs) != len(func_names):
        raise ValueError(f"Length of 'funcs' (got {len(funcs)}) and "
                         f"'func_names' (got {len(func_names)}) must be equal")

    # Zero-copy `args` for each function (if necessary):
    args_list = [args] * len(funcs) if not isinstance(args, list) else args

    iterator = zip(funcs, args_list, func_names)
    if progress_bar:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(funcs))

    ts: List[List[str]] = []
    for func, inputs, name in iterator:
        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            args = inputs() if callable(inputs) else inputs
            args = require_grad(args, backward)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = func(*args)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if backward:
                if isinstance(out, (tuple, list)):
                    out = sum(o.sum() for o in out if isinstance(o, Tensor))
                elif isinstance(out, dict):
                    out = out.values()
                    out = sum(o.sum() for o in out if isinstance(o, Tensor))

                out_grad = torch.randn_like(out)
                t_start = time.perf_counter()

                out.backward(out_grad)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        if per_step:
            ts.append([name, f'{t_forward/num_steps:.6f}s'])
        else:
            ts.append([name, f'{t_forward:.4f}s'])
        if backward:
            if per_step:
                ts[-1].append(f'{t_backward/num_steps:.6f}s')
                ts[-1].append(f'{(t_forward + t_backward)/num_steps:.6f}s')
            else:
                ts[-1].append(f'{t_backward:.4f}s')
                ts[-1].append(f'{t_forward + t_backward:.4f}s')

    header = ['Name', 'Forward']
    if backward:
        header.extend(['Backward', 'Total'])

    print(tabulate(ts, headers=header, tablefmt='psql'))


def get_func_name(func: Callable) -> str:
    if hasattr(func, '__name__'):
        return func.__name__
    elif hasattr(func, '__class__'):
        return func.__class__.__name__
    raise ValueError("Could not infer name for function '{func}'")
