import torch
from torch.utils.data.datapipe import IterDataPipe
from torch.utils.data.datapipes._decorator import functional_datapipe

from torch_geometric.data import Batch


@functional_datapipe('batch_graphs')
class Batcher(torch.utils.data.datapipe.Batcher):
    def __init__(
        self,
        datapipe: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
    ):
        super().__init__(
            datapipe,
            batch_size,
            drop_last,
            wrapper_class=Batch.from_data_list,
        )
