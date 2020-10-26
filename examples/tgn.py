import os.path as osp
from torch_geometric.datasets import JODIEDataset

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

print(train_data)
print(val_data)
print(test_data)
