from typing import Callable, List, Union, Tuple

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class Compose(BaseTransform):
    r"""Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
        interval (Tuple[float, float], optional): A tuple representing the
            interval for the transformation. Defaults to (0.0, 1.0).
    """
    def __init__(self, transforms: List[Callable], interval: Tuple[float, float] = (0.0, 1.0)):
        self.transforms = transforms
        self.interval = interval

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Pass the interval argument to the transformed data
        data.interval = self.interval
        
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
        interval (Tuple[float, float], optional): A tuple representing the
            interval for the transformation. Defaults to (0.0, 1.0).
    """
    def __init__(self, filters: List[Callable], interval: Tuple[float, float] = (0.0, 1.0)):
        self.filters = filters
        self.interval = interval

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> bool:
        # Pass the interval argument to the transformed data
        data.interval = self.interval
        
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
