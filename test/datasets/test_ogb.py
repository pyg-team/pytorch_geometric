from torch_geometric.datasets import OGB_MAG


def test_ogbmag():
    dataset = OGB_MAG('/tmp/OGB')
    print(dataset)
    print(dataset[0])
