from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, InMemoryDataset


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self,
        dataset: Union[Data, Dataset, List[Data]],
        input_nodes: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
    ):

        if isinstance(dataset, Data):
            y = dataset.y.view(-1)
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.data.y.view(-1)
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1. / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)
