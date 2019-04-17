import torch
from torch_geometric.data import Data, InMemoryDataset


def test_in_memory_dataset():
    class TestDataset(InMemoryDataset):
        def __init__(self, data_list):
            super(TestDataset, self).__init__('/tmp/TestDataset')
            self.data, self.slices = self.collate(data_list)

        def _download(self):
            pass

        def _process(self):
            pass

    x = torch.Tensor([[1], [1], [1]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])
    test_int = 1

    data1 = Data(x=x, edge_index=edge_index, face=face, test_int=test_int)
    data1.num_nodes = 10

    data2 = Data(x=x, edge_index=edge_index, face=face, test_int=test_int)
    data2.num_nodes = 5

    dataset = TestDataset([data1, data2])
    assert len(dataset) == 2
    assert dataset[0].num_nodes == 10
    assert dataset[1].num_nodes == 5
