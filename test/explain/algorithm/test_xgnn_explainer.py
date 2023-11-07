import pytest
import torch

from torch_geometric.explain import Explainer, XGNNExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

# from torch_geometric.nn import AttentiveFP, ChebConv, GCNConv, global_add_pool