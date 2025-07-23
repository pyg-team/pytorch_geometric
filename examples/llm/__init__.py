"""PyTorch Geometric LLM examples.

Graph-based retrieval and conversation utilities for PyG.
"""

# WHG-Retriever components
try:
    from .whg_model import WarehouseGRetriever, WarehouseTaskHead  # noqa: F401
    from .whg_demo import (  # noqa: F401
        WarehouseConversationSystem, WHGConversationSystem)
    __all__ = [
        'WarehouseGRetriever',
        'WarehouseTaskHead',
        'WarehouseConversationSystem',
        'WHGConversationSystem',
    ]
except ImportError:
    # Graceful fallback if dependencies not available
    __all__ = []
