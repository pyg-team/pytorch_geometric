from typing import Iterator, List, Optional

import torch

from torch_geometric.data import Dataset


class DynamicBatchSampler(torch.utils.data.sampler.Sampler):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges). When data samples have a
    wide range in sizes, specifying a mini-batch size in terms of number of
    samples is not ideal and can cause CUDA OOM errors.

    Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
    ambiguous, depending on the order of the samples. By default the
    :meth:`__len__` will be undefined. This is fine for most cases but
    progress bars will be infinite. Alternatively, :obj:`num_steps` can be
    supplied to cap the number of mini-batches produced by the sampler.

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num_nodes (int): Size of mini-batch to aim for in number of nodes.
        max_num_edges (int): Size of mini-batch to aim for in number of edges.
        shuffle (bool, optional): If set to :obj:`True`, will have the data
            reshuffled at every epoch. (default: :obj:`False`)
        skip_too_big (bool, optional): If set to :obj:`True`, skip samples
            which cannot fit in a batch by itself. (default: :obj:`False`)
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying examples, but :meth:`__len__` will be :obj:`None` since
            it is ambiguous. (default: :obj:`None`)
    """
    def __init__(
        self,
        dataset: Dataset,
        max_num_nodes: int,
        max_num_edges: int,
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
    ):
        if max_num_nodes <= 0:
            raise ValueError(f"`max_num_nodes` should be a positive integer "
                             f"value (got {max_num_nodes})")
        if max_num_edges <= 0:
            raise ValueError(f"`max_num_edges` should be a positive integer "
                             f"value (got {max_num_edges})")
        self.dataset = dataset
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.max_steps = num_steps or len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = range(len(self.dataset))

        samples: List[int] = []
        current_num_nodes: int = 0
        current_num_edges: int = 0
        num_steps: int = 0
        num_processed: int = 0

        while (num_processed < len(self.dataset)
               and num_steps < self.max_steps):

            for i in indices[num_processed:]:
                data = self.dataset[i]
                num_nodes = data.num_nodes
                num_edges = data.num_edges
                # Check if adding a single graph will cause the mini-batch to
                # more nodes/edges than the maximums given.
                if (current_num_nodes + num_nodes
                        > self.max_num_nodes) or (current_num_edges + num_edges
                                                  > self.max_num_edges):

                    if current_num_nodes == 0 or current_num_edges == 0:
                        if self.skip_too_big:
                            continue
                    else:  # Mini-batch filled:
                        break

                samples.append(i)
                num_processed += 1
                current_num_nodes += num_nodes
                current_num_edges += num_edges

            yield samples
            samples: List[int] = []
            current_num_nodes = 0
            current_num_edges = 0
            num_steps += 1

    def __len__(self) -> int:
        if self.num_steps is None:
            raise ValueError(f"The length of '{self.__class__.__name__}' is "
                             f"undefined since the number of steps per epoch "
                             f"is ambiguous. Either specify `num_steps` or "
                             f"use a static batch sampler.")

        return self.num_steps
