def test_suite_sparse_dataset(get_dataset):
    dataset = get_dataset(group='DIMACS10', name='citationCiteseer')
    assert str(dataset) == ('SuiteSparseMatrixCollection('
                            'group=DIMACS10, name=citationCiteseer)')
    assert len(dataset) == 1


def test_illc1850_suite_sparse_dataset(get_dataset):
    dataset = get_dataset(group='HB', name='illc1850')
    assert str(dataset) == ('SuiteSparseMatrixCollection('
                            'group=HB, name=illc1850)')
    assert len(dataset) == 1
