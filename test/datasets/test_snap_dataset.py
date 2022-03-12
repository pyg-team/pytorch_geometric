from torch_geometric.testing import withDataset


@withDataset(name='ego-facebook')
def test_ego_facebook_snap_dataset(dataset):
    assert str(dataset) == 'SNAP-ego-facebook(10)'
    assert len(dataset) == 10


@withDataset(name='soc-Slashdot0811')
def test_soc_slashdot_snap_dataset(dataset):
    assert str(dataset) == 'SNAP-soc-slashdot0811(1)'
    assert len(dataset) == 1


@withDataset(name='wiki-vote')
def test_wiki_vote_snap_dataset(dataset):
    assert str(dataset) == 'SNAP-wiki-vote(1)'
    assert len(dataset) == 1
