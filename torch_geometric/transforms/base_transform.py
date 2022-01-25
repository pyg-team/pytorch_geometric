from typing import Any


class BaseTransform:
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`~torch_geometric.data.Data` objects, either by implicitly passing
    them as an argument to a :class:`~torch_geometric.data.Dataset`, or by
    applying them explicitly to individual :class:`~torch_geometric.data.Data`
    objects.

    .. code-block:: python

        import torch_geometric.transforms as T
        from torch_geometric.datasets import TUDataset

        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

        dataset = TUDataset(path, name='MUTAG', transform=transform)
        data = dataset[0]  # Implicitly transform data on every access.

        data = TUDataset(path, name='MUTAG')[0]
        data = transform(data)  # Explicitly transform data.
    """
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
