import functools
import inspect
import warnings
from typing import Optional


def deprecated(details: Optional[str] = None, func_name: Optional[str] = None):
    def decorator(func):
        if inspect.isclass(func):
            cls_name = func_name or func.__name__
            func.__init__ = deprecated(details, cls_name)(func.__init__)
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = f"'{func_name or func.__name__}' is deprecated" + (
                f', {details}' if details is not None else '')
            warnings.warn(out)
            return func(*args, **kwargs)

        return wrapper

    return decorator
