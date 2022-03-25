from typing import Optional

import torch
import torchdata
from torchdata.datapipes.iter import IterDataPipe

from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles


@torchdata.datapipes.functional_datapipe('batch_graphs')
class Batcher(torchdata.datapipes.iter.Batcher):
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


@torchdata.datapipes.functional_datapipe('cache')
class Cacher(IterDataPipe):
    def __init__(self, dp: IterDataPipe):
        super().__init__()
        self.dp = dp
        self.cache_list = []
        self.cache_full = False

    def __iter__(self):
        if self.cache_full:
            for d in self.cache_list:
                yield d
        else:
            self.cache_list = []
            for d in self.dp:
                self.cache_list.append(d)
                yield d
            self.cache_full = True


@torchdata.datapipes.functional_datapipe('parse_smiles')
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

    def __iter__(self):
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
