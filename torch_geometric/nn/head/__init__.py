from .node import IdentityNodeHead, MLPNodeHead
from .link import ConcatLinkHead, InnerProductLinkHead
from .graph import MLPGraphHead

__all__ = [
    'IdentityNodeHead',
    'MLPNodeHead',
    'ConcatLinkHead',
    'InnerProductLinkHead',
    'MLPGraphHead',
]

classes = __all__
