import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset


class MyTestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('/tmp/MyTestDataset')
        self.data, self.slices = self.collate(data_list)


def test_in_memory_dataset():
    x1 = torch.Tensor([[1], [1], [1]])
    x2 = torch.Tensor([[2], [2], [2]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])

    data1 = Data(x1, edge_index, face=face, test_int=1, test_str='1')
    data1.num_nodes = 10

    data2 = Data(x2, edge_index, face=face, test_int=2, test_str='2')
    data2.num_nodes = 5

    dataset = MyTestDataset([data1, data2])
    assert str(dataset) == 'MyTestDataset(2)'
    assert len(dataset) == 2

    assert len(dataset[0]) == 6
    assert dataset[0].num_nodes == 10
    assert dataset[0].x.tolist() == x1.tolist()
    assert dataset[0].edge_index.tolist() == edge_index.tolist()
    assert dataset[0].face.tolist() == face.tolist()
    assert dataset[0].test_int == 1
    assert dataset[0].test_str == '1'

    assert len(dataset[1]) == 6
    assert dataset[1].num_nodes == 5
    assert dataset[1].x.tolist() == x2.tolist()
    assert dataset[1].edge_index.tolist() == edge_index.tolist()
    assert dataset[1].face.tolist() == face.tolist()
    assert dataset[1].test_int == 2
    assert dataset[1].test_str == '2'


def test_collate_with_new_dimension():
    class MyData(Data):
        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == 'foo':
                return None
            else:
                return super().__cat_dim__(key, value, *args, **kwargs)

    x = torch.tensor([1, 2, 3], dtype=torch.float)
    foo = torch.randn(4)
    y = torch.tensor(1)

    data = MyData(x=x, foo=foo, y=y)

    dataset = MyTestDataset([data, data])
    assert str(dataset) == 'MyTestDataset(2)'
    assert len(dataset) == 2

    data1 = dataset[0]
    assert len(data1) == 3
    assert data1.x.tolist() == x.tolist()
    assert data1.foo.tolist() == foo.tolist()
    assert data1.y.tolist() == [1]

    data2 = dataset[0]
    assert len(data2) == 3
    assert data2.x.tolist() == x.tolist()
    assert data2.foo.tolist() == foo.tolist()
    assert data2.y.tolist() == [1]


def test_hetero_in_memory_dataset():
    data1 = HeteroData()
    data1.y = torch.randn(5)
    data1['paper'].x = torch.randn(10, 16)
    data1['paper', 'paper'].edge_index = torch.randint(0, 10, (2, 30)).long()

    data2 = HeteroData()
    data2.y = torch.randn(5)
    data2['paper'].x = torch.randn(10, 16)
    data2['paper', 'paper'].edge_index = torch.randint(0, 10, (2, 30)).long()

    dataset = MyTestDataset([data1, data2])
    assert str(dataset) == 'MyTestDataset(2)'
    assert len(dataset) == 2

    assert len(dataset[0]) == 3
    assert dataset[0].y.tolist() == data1.y.tolist()
    assert dataset[0]['paper'].x.tolist() == data1['paper'].x.tolist()
    assert (dataset[0]['paper', 'paper'].edge_index.tolist() == data1[
        'paper', 'paper'].edge_index.tolist())

    assert len(dataset[1]) == 3
    assert dataset[1].y.tolist() == data2.y.tolist()
    assert dataset[1]['paper'].x.tolist() == data2['paper'].x.tolist()
    assert (dataset[1]['paper', 'paper'].edge_index.tolist() == data2[
        'paper', 'paper'].edge_index.tolist())
