import pytest
import torch

from torch_geometric.datasets import GMD
from torch_geometric.testing import onlyOnline


def test_gmd_tasks():
    assert len(GMD.molecules) == len(set(GMD.molecules)) == 140
    for train, ood in GMD.tasks.values():
        assert set(train) <= set(GMD.molecules)
        assert set(ood) <= set(GMD.molecules)
        assert set(train).isdisjoint(ood)

    # Arguments are validated before anything is downloaded:
    with pytest.raises(ValueError, match="Unknown task"):
        GMD('/dev/null', task='fragment_unknown')
    with pytest.raises(ValueError, match="Unknown split"):
        GMD('/dev/null', task='fragment_fusion', split='test')
    with pytest.raises(ValueError, match="does not provide pre-defined"):
        GMD('/dev/null', task='all', split='train')


@onlyOnline
@pytest.mark.dataset
def test_gmd(tmp_path):
    sizes = {'train': 5000, 'val': 1000, 'test_id': 1000, 'test_ood': 1400}
    for split, size in sizes.items():
        dataset = GMD(tmp_path, task='fragment_chain_extension', split=split)
        assert len(dataset) == size
        assert str(dataset) == (f"GMD({size}, task='fragment_chain_extension'"
                                f", split='{split}')")

    assert len(list((tmp_path / 'raw' / 'Alkanes').iterdir())) == 12

    data = dataset[0]
    assert data.molecule == 'C7H16' and data.family == 'Alkanes'
    assert data.z.size() == (23, ) and int((data.z == 6).sum()) == 7
    assert data.pos.size() == data.force.size() == (23, 3)
    assert data.energy.size() == (1, ) and data.frame.tolist() == [1400]
    frame = dataset._data.frame
    assert frame.min() == 1400 and frame.max() == 1599
    assert torch.isfinite(dataset._data.energy).all()

    atomref = dataset.atomref()
    assert atomref.size() == (100, 1)
    atomization_energy = data.energy - atomref[data.z].sum()
    assert -4.5 < float(atomization_energy) / data.num_nodes < -3.5
