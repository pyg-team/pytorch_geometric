from torch_geometric.testing import withDataset


@withDataset(name='DIMACS10/citationCiteseer')
def test_suite_sparse_dataset(get_dataset):
    dataset = get_dataset()
    assert str(dataset) == ('SuiteSparseMatrixCollection('
                            'group=DIMACS10, name=citationCiteseer)')
    assert len(dataset) == 1


@withDataset(name='HB/illc1850')
def test_illc1850_suite_sparse_dataset(get_dataset):
    dataset = get_dataset()
    assert str(dataset) == ('SuiteSparseMatrixCollection('
                            'group=HB, name=illc1850)')
    assert len(dataset) == 1
