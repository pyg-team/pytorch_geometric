import copy

import pytest
import torch

from torch_geometric import EdgeIndex, Index
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.datasets import KarateClub
from torch_geometric.testing import withPackage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor


class MyTestDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__(None, transform=transform)
        self.data, self.slices = self.collate(data_list)


class MyStoredTestDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0], data_cls=data_list[0].__class__)

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])


def test_in_memory_dataset():
    x1 = torch.tensor([[1.0], [1.0], [1.0]])
    x2 = torch.tensor([[2.0], [2.0], [2.0]])
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

    with pytest.warns(UserWarning, match="internal storage format"):
        dataset.data

    assert torch.equal(dataset.x, torch.cat([x1, x2], dim=0))
    assert dataset.edge_index.tolist() == [
        [0, 1, 1, 2, 10, 11, 11, 12],
        [1, 0, 2, 1, 11, 10, 12, 11],
    ]
    assert torch.equal(dataset[1:].x, x2)


def test_stored_in_memory_dataset(tmp_path):
    x1 = torch.tensor([[1.0], [1.0], [1.0]])
    x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data1 = Data(x1, edge_index, num_nodes=3, test_int=1, test_str='1')
    data2 = Data(x2, edge_index, num_nodes=4, test_int=2, test_str='2')

    dataset = MyStoredTestDataset(tmp_path, [data1, data2])
    assert dataset._data.num_nodes == 7
    assert dataset._data._num_nodes == [3, 4]

    assert torch.equal(dataset[0].x, x1)
    assert torch.equal(dataset[0].edge_index, edge_index)
    assert dataset[0].num_nodes == 3
    assert torch.equal(dataset[0].test_int, torch.tensor([1]))
    assert dataset[0].test_str == '1'

    assert torch.equal(dataset[1].x, x2)
    assert torch.equal(dataset[1].edge_index, edge_index)
    assert dataset[1].num_nodes == 4
    assert torch.equal(dataset[1].test_int, torch.tensor([2]))
    assert dataset[1].test_str == '2'


def test_stored_hetero_in_memory_dataset(tmp_path):
    x1 = torch.tensor([[1.0], [1.0], [1.0]])
    x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]])

    data1 = HeteroData()
    data1['paper'].x = x1
    data1['paper'].num_nodes = 3

    data2 = HeteroData()
    data2['paper'].x = x2
    data2['paper'].num_nodes = 4

    dataset = MyStoredTestDataset(tmp_path, [data1, data2])
    assert dataset._data['paper'].num_nodes == 7
    assert dataset._data['paper']._num_nodes == [3, 4]

    assert torch.equal(dataset[0]['paper'].x, x1)
    assert dataset[0]['paper'].num_nodes == 3

    assert torch.equal(dataset[1]['paper'].x, x2)
    assert dataset[1]['paper'].num_nodes == 4


def test_index(tmp_path):
    index1 = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    index2 = Index([0, 1, 1, 2, 2, 3], dim_size=4, is_sorted=True)

    data1 = Data(batch=index1)
    data2 = Data(batch=index2)

    dataset = MyTestDataset([data1, data2])
    assert len(dataset) == 2
    for data, index in zip(dataset, [index1, index2]):
        assert isinstance(data.batch, Index)
        assert data.batch.equal(index)
        assert data.batch.dim_size == index.dim_size
        assert data.batch.is_sorted == index.is_sorted

    dataset = MyStoredTestDataset(tmp_path, [data1, data2])
    assert len(dataset) == 2
    for data, index in zip(dataset, [index1, index2]):
        assert isinstance(data.batch, Index)
        assert data.batch.equal(index)
        assert data.batch.dim_size == index.dim_size
        assert data.batch.is_sorted == index.is_sorted


def test_edge_index(tmp_path):
    edge_index1 = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
    )
    edge_index2 = EdgeIndex(
        [[1, 0, 2, 1, 3, 2], [0, 1, 1, 2, 2, 3]],
        sparse_size=(4, 4),
        sort_order='col',
    )

    data1 = Data(edge_index=edge_index1)
    data2 = Data(edge_index=edge_index2)

    dataset = MyTestDataset([data1, data2])
    assert len(dataset) == 2
    for data, edge_index in zip(dataset, [edge_index1, edge_index2]):
        assert isinstance(data.edge_index, EdgeIndex)
        assert data.edge_index.equal(edge_index)
        assert data.edge_index.sparse_size() == edge_index.sparse_size()
        assert data.edge_index.sort_order == edge_index.sort_order
        assert data.edge_index.is_undirected == edge_index.is_undirected

    dataset = MyStoredTestDataset(tmp_path, [data1, data2])
    assert len(dataset) == 2
    for data, edge_index in zip(dataset, [edge_index1, edge_index2]):
        assert isinstance(data.edge_index, EdgeIndex)
        assert data.edge_index.equal(edge_index)
        assert data.edge_index.sparse_size() == edge_index.sparse_size()
        assert data.edge_index.sort_order == edge_index.sort_order
        assert data.edge_index.is_undirected == edge_index.is_undirected


def test_in_memory_num_classes():
    dataset = MyTestDataset([Data(), Data()])
    assert dataset.num_classes == 0

    dataset = MyTestDataset([Data(y=0), Data(y=1)])
    assert dataset.num_classes == 2

    dataset = MyTestDataset([Data(y=1.5), Data(y=2.5), Data(y=3.5)])
    with pytest.warns(UserWarning, match="unique elements"):
        assert dataset.num_classes == 3

    dataset = MyTestDataset([
        Data(y=torch.tensor([[0, 1, 0, 1]])),
        Data(y=torch.tensor([[1, 0, 0, 0]])),
        Data(y=torch.tensor([[0, 0, 1, 0]])),
    ])
    assert dataset.num_classes == 4

    # Test when `__getitem__` returns a tuple of data objects.
    def transform(data):
        copied_data = copy.copy(data)
        copied_data.y += 1
        return data, copied_data, 'foo'

    dataset = MyTestDataset([Data(y=0), Data(y=1)], transform=transform)
    assert dataset.num_classes == 3


def test_in_memory_dataset_copy():
    data_list = [Data(x=torch.randn(5, 16)) for _ in range(4)]
    dataset = MyTestDataset(data_list)

    copied_dataset = dataset.copy()
    assert id(copied_dataset) != id(dataset)

    assert len(copied_dataset) == len(dataset) == 4
    for copied_data, data in zip(copied_dataset, dataset):
        assert torch.equal(copied_data.x, data.x)

    copied_dataset = dataset.copy([1, 2])
    assert len(copied_dataset) == 2
    assert torch.equal(copied_dataset[0].x, data_list[1].x)
    assert torch.equal(copied_dataset[1].x, data_list[2].x)


def test_to_datapipe():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    dataset = MyTestDataset([data, data])

    dp = dataset.to_datapipe()

    assert isinstance(dp, torch.utils.data.IterDataPipe)
    assert len(dp) == 2

    assert torch.equal(dataset[0].x, list(dp)[0].x)
    assert torch.equal(dataset[0].edge_index, list(dp)[0].edge_index)
    assert torch.equal(dataset[1].x, list(dp)[1].x)
    assert torch.equal(dataset[1].edge_index, list(dp)[1].edge_index)


@withPackage('torch_sparse')
def test_in_memory_sparse_tensor_dataset():
    x = torch.randn(11, 16)
    adj = SparseTensor(
        row=torch.tensor([4, 1, 3, 2, 2, 3]),
        col=torch.tensor([1, 3, 2, 3, 3, 2]),
        sparse_sizes=(11, 11),
    )
    data = Data(x=x, adj=adj)

    dataset = MyTestDataset([data, data])
    assert len(dataset) == 2
    assert torch.allclose(dataset[0].x, x)
    assert dataset[0].adj.sparse_sizes() == (11, 11)
    assert torch.allclose(dataset[1].x, x)
    assert dataset[1].adj.sparse_sizes() == (11, 11)


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


def test_override_behavior():
    class DS1(InMemoryDataset):
        def __init__(self):
            self.enter_download = False
            self.enter_process = False
            super().__init__()

        def _download(self):
            self.enter_download = True

        def _process(self):
            self.enter_process = True

        def download(self):
            pass

        def process(self):
            pass

    class DS2(InMemoryDataset):
        def __init__(self):
            self.enter_download = False
            self.enter_process = False
            super().__init__()

        def _download(self):
            self.enter_download = True

        def _process(self):
            self.enter_process = True

        def process(self):
            pass

    class DS3(InMemoryDataset):
        def __init__(self):
            self.enter_download = False
            self.enter_process = False
            super().__init__()

        def _download(self):
            self.enter_download = True

        def _process(self):
            self.enter_process = True

    class DS4(DS1):
        pass

    ds = DS1()
    assert ds.enter_download
    assert ds.enter_process

    ds = DS2()
    assert not ds.enter_download
    assert ds.enter_process

    ds = DS3()
    assert not ds.enter_download
    assert not ds.enter_process

    ds = DS4()
    assert ds.enter_download
    assert ds.enter_process


def test_lists_of_tensors_in_memory_dataset():
    def tr(n, m):
        return torch.rand((n, m))

    d1 = Data(xs=[tr(4, 3), tr(11, 4), tr(1, 2)])
    d2 = Data(xs=[tr(5, 3), tr(14, 4), tr(3, 2)])
    d3 = Data(xs=[tr(6, 3), tr(15, 4), tr(2, 2)])
    d4 = Data(xs=[tr(4, 3), tr(16, 4), tr(1, 2)])

    data_list = [d1, d2, d3, d4]

    dataset = MyTestDataset(data_list)
    assert len(dataset) == 4
    assert dataset[0].xs[1].size() == (11, 4)
    assert dataset[0].xs[2].size() == (1, 2)
    assert dataset[1].xs[0].size() == (5, 3)
    assert dataset[2].xs[1].size() == (15, 4)
    assert dataset[3].xs[1].size() == (16, 4)


@withPackage('torch_sparse')
def test_lists_of_sparse_tensors():
    e1 = torch.tensor([[4, 1, 3, 2, 2, 3], [1, 3, 2, 3, 3, 2]])
    e2 = torch.tensor([[0, 1, 4, 7, 2, 9], [7, 2, 2, 1, 4, 7]])
    e3 = torch.tensor([[3, 5, 1, 2, 3, 3], [5, 0, 2, 1, 3, 7]])
    e4 = torch.tensor([[0, 1, 9, 2, 0, 3], [1, 1, 2, 1, 3, 2]])
    adj1 = SparseTensor.from_edge_index(e1, sparse_sizes=(11, 11))
    adj2 = SparseTensor.from_edge_index(e2, sparse_sizes=(22, 22))
    adj3 = SparseTensor.from_edge_index(e3, sparse_sizes=(12, 12))
    adj4 = SparseTensor.from_edge_index(e4, sparse_sizes=(15, 15))

    d1 = Data(adj_test=[adj1, adj2])
    d2 = Data(adj_test=[adj3, adj4])

    data_list = [d1, d2]
    dataset = MyTestDataset(data_list)
    assert len(dataset) == 2
    assert dataset[0].adj_test[0].sparse_sizes() == (11, 11)
    assert dataset[0].adj_test[1].sparse_sizes() == (22, 22)
    assert dataset[1].adj_test[0].sparse_sizes() == (12, 12)
    assert dataset[1].adj_test[1].sparse_sizes() == (15, 15)


def test_file_names_as_property_and_method():
    class MyTestDataset(InMemoryDataset):
        def __init__(self):
            super().__init__('/tmp/MyTestDataset')

        @property
        def raw_file_names(self):
            return ['test_file']

        def download(self):
            pass

    MyTestDataset()

    class MyTestDataset(InMemoryDataset):
        def __init__(self):
            super().__init__('/tmp/MyTestDataset')

        def raw_file_names(self):
            return ['test_file']

        def download(self):
            pass

    MyTestDataset()


@withPackage('sqlite3')
def test_to_on_disk_dataset(tmp_path):
    class MyTransform(BaseTransform):
        def forward(self, data: Data) -> Data:
            data.z = 'test_str'
            return data

    in_memory_dataset = KarateClub(transform=MyTransform())

    with pytest.raises(ValueError, match="root directory of 'KarateClub'"):
        in_memory_dataset.to_on_disk_dataset()

    on_disk_dataset = in_memory_dataset.to_on_disk_dataset(tmp_path, log=False)
    assert str(on_disk_dataset) == 'OnDiskKarateClub()'
    assert on_disk_dataset.schema == {
        'x': dict(dtype=torch.float32, size=(-1, 34)),
        'edge_index': dict(dtype=torch.int64, size=(2, -1)),
        'y': dict(dtype=torch.int64, size=(-1, )),
        'train_mask': dict(dtype=torch.bool, size=(-1, )),
    }
    assert in_memory_dataset.transform == on_disk_dataset.transform

    data1 = in_memory_dataset[0]
    data2 = on_disk_dataset[0]

    assert len(data1) == len(data2)
    assert torch.allclose(data1.x, data2.x)
    assert torch.equal(data1.edge_index, data2.edge_index)
    assert torch.equal(data1.y, data2.y)
    assert torch.equal(data1.train_mask, data2.train_mask)
    assert data1.z == data2.z

    on_disk_dataset.close()
