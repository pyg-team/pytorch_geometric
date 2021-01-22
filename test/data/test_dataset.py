import torch
from torch_geometric.data import Data, InMemoryDataset


class DatasetFromList(InMemoryDataset):
    def __init__(self, data_list):
        super(DatasetFromList, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)


def test_in_memory_dataset():
    x = torch.Tensor([[1], [1], [1]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])
    i = 1
    s = '1'

    data1 = Data(x=x, edge_index=edge_index, face=face, test_int=i, test_str=s)
    data1.num_nodes = 10

    data2 = Data(x=x, edge_index=edge_index, face=face, test_int=i, test_str=s)
    data2.num_nodes = 5

    dataset = DatasetFromList([data1, data2])
    assert len(dataset) == 2
    assert dataset[0].num_nodes == 10
    assert len(dataset[0]) == 5
    assert dataset[1].num_nodes == 5
    assert len(dataset[1]) == 5


def test_newaxis_batching():
    class MyData(Data):
        def __cat_dim__(self, key, item):
            if key == 'newaxis_field':
                return None
            else:
                return super().__cat_dim__(key, item)

    N_datas = 3
    N_nodes = torch.randint(low=3, high=10, size=(N_datas,))
    N_features = 8
    target_shape = (4, 7)
    datas = [
        MyData(
            x=torch.randn(n_node, N_features),
            newaxis_field=torch.randn(target_shape)
        )
        for n_node in N_nodes
    ]

    dataset = DatasetFromList(datas)

    assert len(dataset) == N_datas
    for orig, from_dataset in zip(datas, dataset):
        assert orig.x.shape == from_dataset.x.shape
        assert torch.all(orig.x == from_dataset.x)
        assert orig.newaxis_field.shape == from_dataset.newaxis_field.shape
        assert torch.all(orig.newaxis_field == from_dataset.newaxis_field)
