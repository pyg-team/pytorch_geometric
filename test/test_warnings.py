import warnings
from unittest.mock import patch

import pytest

from torch_geometric.warnings import warn


def test_warn():
    with pytest.warns(UserWarning, match='test'):
        warn('test')


@patch('torch_geometric.warnings._is_compiling', return_value=True)
def test_no_warn_if_compiling(_):
    """No warning should be raised to avoid graph breaks when compiling."""
    with warnings.catch_warnings():
        warn('test')
