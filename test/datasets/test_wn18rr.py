from torch_geometric.datasets import WordNet18RR

def test_wn18rr():
    dataset = WordNet18RR('.')
    assert len(dataset) == 93003
    assert len(dataset.data.train_mask[0]) == 93003
    assert sum(dataset.data.train_mask[0]) == 86835

