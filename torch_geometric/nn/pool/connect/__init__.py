r"""Graph connection package.

This package provides classes for determining coarsened graph connections in
graph pooling scenarios.
"""

from .base import Connect, ConnectOutput
from .filter_edges import FilterEdges

__all__ = [
    'Connect',
    'ConnectOutput',
    'FilterEdges',
]
