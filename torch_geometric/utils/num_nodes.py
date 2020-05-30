from typing import Optional

import torch


def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes
