from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_ego_facebook_snap_dataset(get_dataset):
    dataset = get_dataset(name='ego-facebook')
    assert str(dataset) == 'SNAP-ego-facebook(10)'
    assert len(dataset) == 10


@onlyOnline
@onlyFullTest
def test_soc_slashdot_snap_dataset(get_dataset):
    dataset = get_dataset(name='soc-Slashdot0811')
    assert str(dataset) == 'SNAP-soc-slashdot0811(1)'
    assert len(dataset) == 1


@onlyOnline
@onlyFullTest
def test_wiki_vote_snap_dataset(get_dataset):
    dataset = get_dataset(name='wiki-vote')
    assert str(dataset) == 'SNAP-wiki-vote(1)'
    assert len(dataset) == 1
