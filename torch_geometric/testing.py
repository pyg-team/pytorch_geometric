import os
from typing import Callable


def onlyFullTest(func: Callable):
    import pytest

    return pytest.mark.skipif(
        os.getenv('FULL_TEST', '0') != '1',
        reason="Skip for fast testing run",
    )(func)


def is_full_test():
    return os.getenv('FULL_TEST', '0') == '1'
