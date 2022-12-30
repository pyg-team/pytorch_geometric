from typing import List, Tuple

import torch
from torch import Tensor


class KGTripletLoader(torch.utils.data.DataLoader):
    def __init__(self, head_index: Tensor, rel_type: Tensor,
                 tail_index: Tensor, **kwargs):
        self.head_index = head_index
        self.rel_type = rel_type
        self.tail_index = tail_index

        super().__init__(range(head_index.numel()), collate_fn=self.sample,
                         **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.head_index.device)

        head_index = self.head_index[index]
        rel_type = self.rel_type[index]
        tail_index = self.tail_index[index]

        return head_index, rel_type, tail_index
