import torchdata
from torchdata.datapipes.iter import IterDataPipe

from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles


@torchdata.datapipes.functional_datapipe('batch_graphs')
class Batcher(torchdata.datapipes.iter.Batcher):
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


@torchdata.datapipes.functional_datapipe('parse_smiles')
class SMILESParser(IterDataPipe):
    def __init__(self, dp: IterDataPipe):
        self.dp = dp

    def __iter__(self):
        for d in self.dp:
            # TODO: str -> from_smiles
            # TODO: Dict -> from_smiles(data['smiles']
            # TODO: NAN values?
            yield from_smiles(d['smiles'])
