
import numpy as np
import torch
import torch.nn.functional as F
# TODO: is this allowed?
from skimage.restoration import denoise_tv_chambolle

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator


class GMixup(GraphGenerator):
    pass
