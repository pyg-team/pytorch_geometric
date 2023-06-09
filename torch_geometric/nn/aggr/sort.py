from typing import Optional

import torch
from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation


class SortAggregation(Aggregation):
    r"""The pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.

    Args:
        k (int): The number of nodes to hold for each graph.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        fill_value = x.detach().min() - 1
        batch_x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                         fill_value=fill_value,
                                         max_num_elements=max_num_elements)
        B, N, D = batch_x.size()

        _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
        arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
        perm = perm + arange.view(-1, 1)

        batch_x = batch_x.view(B * N, D)
        batch_x = batch_x[perm]
        batch_x = batch_x.view(B, N, D)

        if N >= self.k:
            batch_x = batch_x[:, :self.k].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, self.k - N, D), fill_value)
            batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

        batch_x[batch_x == fill_value] = 0
        x = batch_x.view(B, self.k * D)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k})')
