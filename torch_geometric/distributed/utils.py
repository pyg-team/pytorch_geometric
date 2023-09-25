from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from torch_geometric.sampler import SamplerOutput
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
        self.out: Dict[NodeType, np.array] = {}

        for ntype in node_types:
            self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.with_dupl.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.out.update({ntype: np.empty(0, dtype=np.int64)})


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

        if self.disjoint:
            for ntype in node_types:
                self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
                self.with_dupl.update(
                    {ntype: torch.empty(0, dtype=torch.int64)})
                self.out.update({ntype: np.empty(0, dtype=np.int64)})
        else:
            for ntype in node_types:
                self.src.update({ntype: None})
                self.with_dupl.update({ntype: None})
                self.out.update({ntype: None})


def remove_duplicates(
    out: SamplerOutput,
    node: np.array,
    batch: Optional[np.array] = None,
    disjoint: bool = False,
):
    num_node = len(node)
    out_node_numpy = out.node.numpy()
    node_numpy = np.concatenate((node, out_node_numpy))

    if not disjoint:
        _, idx = np.unique(node_numpy, return_index=True)
        node = node_numpy[np.sort(idx)]
        src = torch.from_numpy(node[num_node:])

        return (src, node, None, None)
    else:
        batch_numpy = np.concatenate((batch, out.batch.numpy()))

        disjoint_numpy = np.array((batch_numpy, node_numpy))
        _, idx = np.unique(disjoint_numpy, axis=1, return_index=True)

        batch = disjoint_numpy[0][np.sort(idx)]
        node = disjoint_numpy[1][np.sort(idx)]

        src_batch = torch.from_numpy(batch[num_node:])
        src = torch.from_numpy(node[num_node:])

        return (src, node, src_batch, batch)
