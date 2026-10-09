import runpy
import sys
from pathlib import Path

import pytest

from torch_geometric.testing import withPackage


@withPackage('ogb')
def test_negative_num_workers(monkeypatch, capsys) -> None:
    path = Path(__file__).parents[2] / 'examples' / 'llm' / 'glem.py'
    monkeypatch.setattr(sys, 'argv', [str(path), '--num_workers', '-1'])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(path), run_name='__main__')
    assert exc.value.code == 2
    assert '--num_workers must be non-negative' in capsys.readouterr().err
