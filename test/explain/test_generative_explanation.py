import os.path as osp
from typing import Optional, Union

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import GenerativeExplanation
from torch_geometric.explain.config import MaskType
from torch_geometric.testing import withPackage

