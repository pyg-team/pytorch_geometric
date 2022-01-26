from .node import IdentityNodeHead, MLPNodeHead
from .link import MLPLinkHead, InnerProductLinkHead
from .graph import MLPGraphHead

__all__ = [
    'IdentityNodeHead',
    'MLPNodeHead',
    'MLPLinkHead',
    'InnerProductLinkHead',
    'MLPGraphHead',
]

classes = __all__
