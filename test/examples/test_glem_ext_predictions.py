import os
import runpy
from pathlib import Path

import pytest
import torch

from torch_geometric.testing import withPackage


@pytest.fixture
def glem():
    path = Path(__file__).parents[2] / 'examples' / 'llm' / 'glem.py'
    return runpy.run_path(str(path))


@withPackage('ogb')
@pytest.mark.parametrize('content,expected', [
    (b'<!DOCTYPE html><html>quota</html>', True),
    (b'\xef\xbb\xbf \n<HTML>quota</HTML>', True),
    (b'<checkpoint>user data</checkpoint>', False),
    (b'<htmlish>user data</htmlish>', False),
    (b'\x80\x02pickle data', False),
    (b'', False),
])
def test_html_page(glem, tmp_path, content, expected) -> None:
    path = tmp_path / 'predictions.pt'
    path.write_bytes(content)
    assert glem['is_html_page'](str(path)) is expected
    assert not glem['is_html_page'](str(tmp_path / 'missing.pt'))


@withPackage('ogb')
@pytest.mark.parametrize('legacy', [False, True])
def test_load_ext_predictions_existing(glem, tmp_path, monkeypatch,
                                       legacy) -> None:
    root = tmp_path / 'data' / 'ogb'
    folder = root / 'ext_preds'
    if legacy:
        folder = root / 'ogbn_products' / 'ext_preds'
    folder.mkdir(parents=True)
    path = folder / 'giant_sagn_scr.pt'
    pred = torch.randn(3, 2)
    torch.save(pred, path)
    original = path.read_bytes()
    load = glem['load_ext_predictions']
    monkeypatch.setitem(load.__globals__, 'download_google_url',
                        lambda **kwargs: str(path))
    assert torch.equal(load(str(root), torch.device('cpu')), pred)
    assert path.read_bytes() == original


@withPackage('ogb')
def test_load_ext_predictions_quota_retry(glem, tmp_path, monkeypatch) -> None:
    root = tmp_path / 'data' / 'ogb'
    legacy = root / 'ogbn_products' / 'ext_preds' / 'giant_sagn_scr.pt'
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'<!doctype html>quota')
    path = root / 'ext_preds' / 'giant_sagn_scr.pt'
    pred = torch.randn(3, 2)
    load = glem['load_ext_predictions']

    def download(**kwargs):
        assert root.is_dir()  # Cleanup must not remove the caller's root.
        assert not legacy.parent.parent.exists()
        assert not path.exists()
        os.makedirs(kwargs['folder'], exist_ok=True)
        torch.save(pred, path)
        return str(path)

    monkeypatch.setitem(load.__globals__, 'download_google_url', download)
    assert torch.equal(load(str(root), torch.device('cpu')), pred)


@withPackage('ogb')
@pytest.mark.parametrize('content', [b'<!DOCTYPE html>quota', b'<checkpoint>'])
def test_load_ext_predictions_failed_download(glem, tmp_path, monkeypatch,
                                              content) -> None:
    path = tmp_path / 'ext_preds' / 'giant_sagn_scr.pt'
    load = glem['load_ext_predictions']

    def download(**kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setitem(load.__globals__, 'download_google_url', download)
    if content.startswith(b'<!DOCTYPE'):
        with pytest.raises(RuntimeError, match='daily quota'):
            load(str(tmp_path), torch.device('cpu'))
        assert not path.exists()
    else:
        with pytest.raises(Exception) as exc:
            load(str(tmp_path), torch.device('cpu'))
        assert 'daily quota' not in str(exc.value)
        assert path.read_bytes() == content
