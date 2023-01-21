import functools
import inspect
import warnings
from typing import Optional


def deprecated(details: Optional[str] = None, func_name: Optional[str] = None):
    def decorator(func):
        name = func_name or func.__name__

        if inspect.isclass(func):
            func = type(func.__name__, (func, ), {})
            func.__init__ = deprecated(details, name)(func.__init__)
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = f"'{name}' is deprecated"
            if details is not None:
                out += f", {details}"
            warnings.warn(out)
            return func(*args, **kwargs)

        return wrapper

    return decorator
