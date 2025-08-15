from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_ego_facebook_snap_dataset(get_dataset):
    import warnings

    import torch
    from packaging import version

    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        try:
            from torch.serialization import add_safe_globals

            from torch_geometric.datasets.snap_dataset import EgoData

            add_safe_globals([EgoData])
        except ImportError:
            warnings.warn(
                "add_safe_globals is expected but not found in "
                "torch.serialization.", stacklevel=2)
    else:
        warnings.warn(
            "add_safe_globals is not available in this version "
            "of PyTorch; continuing without it.", stacklevel=2)

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
