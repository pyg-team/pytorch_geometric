from typing import List, Tuple

import torch
from torch import Tensor


class KGTripletLoader(torch.utils.data.DataLoader):
    def __init__(self, head: Tensor, rel: Tensor, tail: Tensor, **kwargs):
        self.head = head
        self.rel = rel
        self.tail = tail
        super().__init__(range(head.numel()), collate_fn=self.sample, **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.head.device)
        return self.head[index], self.rel[index], self.tail[index]
