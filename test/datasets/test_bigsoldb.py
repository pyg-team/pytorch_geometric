import os
import os.path as osp

import pytest
import torch

from torch_geometric.datasets import BigSolDB

# Minimal fake CSV matching BigSolDB column schema
FAKE_CSV = (
    "SMILES_Solute,Temperature_K,Solvent,SMILES_Solvent,"
    "Solubility(mole_fraction),Solubility(mol/L),LogS(mol/L),"
    "Compound_Name,CAS,PubChem_CID,FDA_Approved,Source\n"
    "CCO,298.15,ethanol,CCO,0.1,1.5,0.176,Ethanol,64-17-5,702,Yes,DOI1\n"
    "c1ccccc1,298.15,toluene,Cc1ccccc1,0.05,0.8,-0.097,Benzene,"
    "71-43-2,241,No,DOI2\n"
    "CC(=O)O,310.0,ethanol,CCO,0.2,2.0,0.301,AceticAcid,"
    "64-19-7,176,No,DOI3\n"
    "CCN,298.15,water,O,0.3,3.0,0.477,Ethylamine,"
    "75-04-7,6329,No,DOI4\n")


@pytest.fixture()
def dataset(tmp_path):
    """Create a BigSolDB dataset backed by a fake CSV (no download)."""
    raw_dir = tmp_path / 'BigSolDB' / 'raw'
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / 'BigSolDBv2.1.csv'
    csv_path.write_text(FAKE_CSV)

    return BigSolDB(root=str(tmp_path / 'BigSolDB'))


def test_bigsoldb_len(dataset):
    # 4 rows in fake CSV, all have valid SMILES → 4 graphs
    assert len(dataset) == 4


def test_bigsoldb_data_object(dataset):
    d = dataset[0]

    # solute graph
    assert d.x.dim() == 2
    assert d.x.size(1) == 9  # 9 atom features (from_smiles default)
    assert d.edge_index.size(0) == 2

    # solvent graph
    assert d.x_solvent.dim() == 2
    assert d.x_solvent.size(1) == 9
    assert d.edge_index_solvent.size(0) == 2

    # measurement conditions
    assert d.temperature.shape == torch.Size([1])
    assert isinstance(d.solvent_name, str)
    assert isinstance(d.compound_name, str)

    # target
    assert d.y.shape == torch.Size([1])
    assert d.y.dtype == torch.float


def test_bigsoldb_solvent_filter(tmp_path):
    raw_dir = tmp_path / 'BigSolDB_eth' / 'raw'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'BigSolDBv2.1.csv').write_text(FAKE_CSV)

    eth = BigSolDB(root=str(tmp_path / 'BigSolDB_eth'), solvent='ethanol')
    # fake CSV has 2 ethanol rows (CCO and CC(=O)O)
    assert len(eth) == 2
    for d in eth:
        assert d.solvent_name.lower() == 'ethanol'


def test_bigsoldb_solvent_filter_case_insensitive(tmp_path):
    raw_dir = tmp_path / 'BigSolDB_case' / 'raw'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'BigSolDBv2.1.csv').write_text(FAKE_CSV)

    ds = BigSolDB(root=str(tmp_path / 'BigSolDB_case'), solvent='Ethanol')
    assert len(ds) == 2


def test_bigsoldb_repr(dataset):
    assert 'BigSolDB' in repr(dataset)
    assert str(len(dataset)) in repr(dataset)


def test_bigsoldb_repr_with_solvent(tmp_path):
    raw_dir = tmp_path / 'BigSolDB_repr' / 'raw'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'BigSolDBv2.1.csv').write_text(FAKE_CSV)

    ds = BigSolDB(root=str(tmp_path / 'BigSolDB_repr'), solvent='toluene')
    assert "solvent='toluene'" in repr(ds)


def test_bigsoldb_separate_cache_per_solvent(tmp_path):
    """Different solvent filters must produce different processed files."""
    for subdir in ['raw']:
        d = tmp_path / 'BigSolDB_cache' / subdir
        d.mkdir(parents=True)
    (tmp_path / 'BigSolDB_cache' / 'raw' /
     'BigSolDBv2.1.csv').write_text(FAKE_CSV)

    root = str(tmp_path / 'BigSolDB_cache')
    all_ds = BigSolDB(root=root)
    eth_ds = BigSolDB(root=root, solvent='ethanol')

    assert len(all_ds) != len(eth_ds)

    # processed files must be different
    processed = os.listdir(osp.join(root, 'processed'))
    assert 'data.pt' in processed
    assert 'data_ethanol.pt' in processed


def test_bigsoldb_y_values(dataset):
    """LogS values should match the fake CSV entries."""
    log_s_values = [d.y.item() for d in dataset]
    expected = [0.176, -0.097, 0.301, 0.477]
    for actual, exp in zip(sorted(log_s_values), sorted(expected)):
        assert abs(actual - exp) < 1e-3
