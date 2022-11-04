class StepsLoader():
    r"""A wrapper for :class:`~torch_geometric.loader.NeighborLoader`,
    that allows to handle custom number of steps.

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, StepsLoader

        loader = NeighborLoader(...)
        loader = StepsLoader(loader, num_steps=100)

    Args:
        loader (torch_geometric.loader.NeighborLoader):
            The :class:`~torch_geometric.loader.NeighborLoader` object.
        num_steps (int): The number of batches to iterate over, -1 means
            iterating through all the data.
    """
    def __init__(self, loader, num_steps):
        super(StepsLoader, self).__init__()
        self.loader = loader
        self.iter_loader = iter(loader)
        self.num_steps = num_steps
        self.idx = 0
        self.dataset = loader.dataset
        self.data = loader.data
        self.node_sampler = loader.node_sampler

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_steps

    def __next__(self):
        if self.idx == self.num_steps:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
