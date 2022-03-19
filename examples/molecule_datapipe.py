import torchdata

import torch_geometric.data.datapipes  # noqa

path = '/Users/rusty1s/Downloads/HIV.csv'

datapipe = torchdata.datapipes.iter.FileOpener([path])
datapipe = datapipe.parse_csv_as_dict()
datapipe = datapipe.parse_smiles()
datapipe = datapipe.shuffle()
datapipe = datapipe.batch_graphs(batch_size=128)

for batch in datapipe:
    print(batch)
