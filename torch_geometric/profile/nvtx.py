import torch
from typing import Optional
from functools import wraps

def nvtxit(name: Optional[str] = None, n_warmups: int = 0, n_iters: Optional[int] = None):
    def nvtx(func):

        nonlocal name
        iters_so_far = 0
        if name is None:
            name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal iters_so_far
            if not torch.cuda.is_available():
                return func(*args, **kwargs)
            elif iters_so_far < n_warmups:
                iters_so_far += 1
                return func(*args, **kwargs)
            elif n_iters is None or iters_so_far < n_iters + n_warmups:
                torch.cuda.nvtx.range_push(f"{name}_{iters_so_far}")
                result = func(*args, **kwargs)
                torch.cuda.nvtx.range_pop()
                iters_so_far += 1
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return nvtx
