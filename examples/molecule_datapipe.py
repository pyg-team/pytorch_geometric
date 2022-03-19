import time
from typing import Dict

import torchdata.datapipes as dp

from torch_geometric.utils import from_smiles

path = '/Users/rusty1s/Downloads/HIV.csv'


@dp.functional_datapipe('parse_smiles')
class SMILESParser(dp.iter.IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for d in self.dp:
            # TODO: str -> from_smiles
            # TODO: Dict -> from_smiles(data['smiles']
            # TODO: NAN values?
            yield from_smiles(d['smiles'])


def filter_fn(data: Dict[str, str]) -> bool:
    return len(data) > 0


datapipe = dp.iter.FileOpener([path])
datapipe = datapipe.parse_csv_as_dict()
datapipe = datapipe.parse_smiles()
datapipe = datapipe.shuffle()
datapipe = datapipe.batch_graphs(batch_size=4)
# datapipe = datapipe.filter(filter_fn)
# datapipe = datapipe.hello()

t = time.perf_counter()
for i, bla in enumerate(datapipe):
    print(bla)
    # if i > 5:
    #     break
print(time.perf_counter() - t)

# dp = IterableWrapper([(path, io.StringIO(path))])

# for example in dp.readlines():
#     print(example)
