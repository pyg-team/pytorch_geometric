import warnings


def deprecated(details=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            out = f"'{func.__name__}' is deprecated" + (
                f', {details}' if details is not None else '')
            warnings.warn(out, DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return decorator
