from typing import Any

try:
    import pyg_lib
    _PYG_LIB_AVAILABLE = True
except ImportError:
    _PYG_LIB_AVAILABLE = False


def is_available() -> bool:
    return _PYG_LIB_AVAILABLE


def get() -> Any:
    if _PYG_LIB_AVAILABLE:
        return pyg_lib
    raise ImportError("No module named 'pyg_lib'")
