from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from ordered_set import OrderedSet
from torch import Tensor

from torch_geometric.typing import Dict, NodeType


@dataclass
class NodeDict:
    r""" Class used during hetero sampling. It contains fields that refer to
    NodeDict in three situations:
    1) the nodes are to serve as srcs in the next layer,
    2) nodes with duplicates, that are further needed to create COO output
       matrix,
    3) output nodes without duplicates.
    """
    def __init__(self, node_types):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Union[OrderedSet[List[int]], Dict[NodeType, Tensor]] = {}

        for ntype in node_types:
            self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.with_dupl.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.out.update({ntype: OrderedSet([])})


@dataclass
class BatchDict:
    r""" Class used during disjoint hetero sampling. It contains fields that
    refer to BatchDict in three situations:
    1) the batch is to serve as initial subgraph ids for src nodes in the next
       layer,
    2) subgraph ids with duplicates, that are further needed to create COO
       output matrix,
    3) output subgraph ids without duplicates.
    """
    def __init__(self, node_types, disjoint):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Dict[NodeType, Tensor] = {}
        self.disjoint = disjoint

        if not self.disjoint:
            for ntype in node_types:
                self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
                self.with_dupl.update(
                    {ntype: torch.empty(0, dtype=torch.int64)})
                self.out.update({ntype: torch.empty(0, dtype=torch.int64)})
        else:
            self.src = None
            self.with_dupl = None
            self.out = None
