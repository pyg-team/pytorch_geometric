import os
from typing import Callable


def is_full_test():
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable):
    import pytest
    return pytest.mark.skipif(not is_full_test(), reason="Fast test run")(func)
