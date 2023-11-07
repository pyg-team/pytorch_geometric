import logging
from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel
from torch_geometric.nn.conv.message_passing import MessagePassing