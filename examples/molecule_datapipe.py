from typing import Tuple

import torchdata.datapipes as dp

path = '/Users/rusty1s/Downloads/HIV.csv'


@dp.functional_datapipe('hello')
class MyConverter(dp.iter.IterDataPipe):
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for d in self.dp:
            yield d


def filter_fn(data: Tuple[str, str]) -> bool:
    return len(data[1]) > 0


datapipe = dp.iter.FileOpener([path])
datapipe = datapipe.readlines(skip_lines=1)
datapipe = datapipe.filter(filter_fn)
datapipe = datapipe.hello()

for bla in list(datapipe)[:5]:
    print(bla)

# dp = IterableWrapper([(path, io.StringIO(path))])

# for example in dp.readlines():
#     print(example)
