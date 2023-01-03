import copy
from typing import Any, Callable, Optional, Sequence

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

try:
    from torch.utils.data import IterDataPipe, functional_datapipe
    from torch.utils.data.datapipes.iter import Batcher as IterBatcher
except ImportError:
    IterDataPipe = IterBatcher = object

    def functional_datapipe(name: str) -> Callable:
        return lambda cls: cls


@functional_datapipe('batch_graphs')
class Batcher(IterBatcher):
    def __init__(
        self,
        dp: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
    ):
        super().__init__(
            dp,
            batch_size=batch_size,
            drop_last=drop_last,
            wrapper_class=Batch.from_data_list,
        )


@functional_datapipe('parse_smiles')
class SMILESParser(IterDataPipe):
    def __init__(
        self,
        dp: IterDataPipe,
        smiles_key: str = 'smiles',
        target_key: Optional[str] = None,
    ):
        super().__init__()
        self.dp = dp
        self.smiles_key = smiles_key
        self.target_key = target_key

    def __iter__(self) -> Any:
        for d in self.dp:
            if isinstance(d, str):
                data = from_smiles(d)
            elif isinstance(d, dict):
                data = from_smiles(d[self.smiles_key])
                if self.target_key is not None:
                    y = d.get(self.target_key, None)
                    if y is not None:
                        y = float(y) if len(y) > 0 else float('NaN')
                        data.y = torch.tensor([y], dtype=torch.float)
            else:
                raise ValueError(
                    f"'{self.__class__.__name__}' expected either a string or "
                    f"a dict as input (got '{type(d)}')")

            yield data


class DatasetAdapter(IterDataPipe):
    def __init__(self, dataset: Sequence[Any]):
        super().__init__()
        self.dataset = dataset
        self.range = range(len(self))

    def is_shardable(self) -> bool:
        return True

    def apply_sharding(self, num_shards: int, shard_idx: int):
        self.range = range(shard_idx, len(self), num_shards)

    def __iter__(self) -> Any:
        for i in self.range:
            yield self.dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)


def functional_transform(name: str) -> Callable:
    def wrapper(cls: Any) -> Any:
        @functional_datapipe(name)
        class DynamicMapper(IterDataPipe):
            def __init__(self, dp: IterDataPipe, *args, **kwargs):
                super().__init__()
                self.dp = dp
                self.fn = cls(*args, **kwargs)

            def __iter__(self) -> Any:
                for data in self.dp:
                    yield self.fn(copy.copy(data))

        return cls

    return wrapper
