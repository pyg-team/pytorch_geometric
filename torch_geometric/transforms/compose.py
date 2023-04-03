from typing import Callable, List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class Compose(BaseTransform):
    r"""Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


class ComposeFilters:
    r"""Composes several filters together.

    Args:
        filters (List[Callable]): List of filters to compose.
    """
    def __init__(self, filters: List[Callable]):
        self.filters = filters

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> bool:
        for filter_fn in self.filters:
            if isinstance(data, (list, tuple)):
                if not all([filter_fn(d) for d in data]):
                    return False
            elif not filter_fn(data):
                return False
        return True

    def __repr__(self) -> str:
        args = [f'  {filter_fn}' for filter_fn in self.filters]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
