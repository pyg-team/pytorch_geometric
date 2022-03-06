from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, InMemoryDataset


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).

    **Graph-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, ImbalancedSampler

        sampler = ImbalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset or Data): The dataset from which to sample the data,
            either given as a :class:`~torch_geometric.data.Dataset` or
            :class:`~torch_geometric.data.Data` object.
        input_nodes (Tensor, optional): The indices of nodes that are used by
            the corresponding loader, *e.g.*, by
            :class:`~torch_geometric.loader.NeighborLoader`.
            If set to :obj:`None`, all nodes will be considered.
            This argument should only be set for node-level loaders and does
            not have any effect when operating on a set of graphs as given by
            :class:`~torch_geometric.data.Dataset`. (default: :obj:`None`)
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to :obj:`None`, will sample as much elements as there
            exists in the underlying data. (default: :obj:`None`)
    """
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
