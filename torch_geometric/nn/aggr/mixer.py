from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.models.graph_mixer import _MLPMixer


class MixerAggregation(Aggregation):
    def __init__(self) -> None:
        self.mlp_mixer = _MLPMixer(
            num_tokens=3,
            in_channels=5,
            out_channels=7,
            dropout=0.5,
        )

    def reset_parameters(self):
        self.mlp_mixer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=self.max_num_elements)

        return self.mlp_mixer()
