import functools
import inspect
import warnings
from typing import Any, Callable, Optional


def deprecated(
    details: Optional[str] = None,
    func_name: Optional[str] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__

        if inspect.isclass(func):
            cls = type(func.__name__, (func, ), {})
            cls.__init__ = deprecated(details, name)(  # type: ignore
                func.__init__)
            cls.__doc__ = func.__doc__
            return cls

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = f"'{name}' is deprecated"
            if details is not None:
                out += f", {details}"
            warnings.warn(out)
            return func(*args, **kwargs)

        return wrapper

    return decorator
