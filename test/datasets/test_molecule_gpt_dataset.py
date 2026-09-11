import json
import os
import time
import warnings
from typing import Any, List

import pytest
import requests

from torch_geometric.datasets import MoleculeGPTDataset
from torch_geometric.datasets.molecule_gpt_dataset import _get_pubchem_json
from torch_geometric.testing import withPackage

BUSY = {
    'Fault': {
        'Code': 'PUGVIEW.ServerBusy',
        'Message': 'Too many requests or server too busy',
    }
}
NOT_FOUND = {'Fault': {'Code': 'PUGVIEW.NotFound', 'Message': 'No data'}}
PAGE = {'Annotations': {'Page': 1, 'Annotation': []}}


class _Response:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = 'OK' if self.ok else 'Error'
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError('Expecting value')
        return self._payload


def _mock_pubchem(monkeypatch, responses: List[Any]):
    calls: List[str] = []
    sleeps: List[float] = []

    def get(url, timeout):
        calls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    return calls, sleeps


def test_get_pubchem_json_retries_transient_errors(monkeypatch):
    calls, sleeps = _mock_pubchem(monkeypatch, [
        _Response(503, BUSY),
        requests.ConnectionError('conn reset'),
        requests.ReadTimeout('read timed out'),
        requests.exceptions.ChunkedEncodingError('IncompleteRead'),
        requests.exceptions.ContentDecodingError('bad gzip'),
        _Response(502, '<html>Bad Gateway</html>'),
        _Response(200, PAGE),
    ])
    out = _get_pubchem_json('url', num_retries=6, delay=1.0)
    assert out == PAGE
    assert calls == ['url'] * 7
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_get_pubchem_json_raises_after_retries(monkeypatch):
    calls, sleeps = _mock_pubchem(monkeypatch, [_Response(503, BUSY)] * 3)
    with pytest.raises(RuntimeError, match='PUGVIEW.ServerBusy.*3 attempts'):
        _get_pubchem_json('url', num_retries=2)
    assert len(calls) == 3
    assert len(sleeps) == 2

    calls, _ = _mock_pubchem(monkeypatch, [])
    with pytest.raises(ValueError, match='non-negative'):
        _get_pubchem_json('url', num_retries=-1)
    assert calls == []


@pytest.mark.parametrize('response', [
    _Response(404, NOT_FOUND),
    _Response(400, '<html>Bad Request</html>'),
    _Response(200, NOT_FOUND),
])
def test_get_pubchem_json_fails_fast(monkeypatch, response):
    calls, sleeps = _mock_pubchem(monkeypatch, [response])
    match = f'HTTP {response.status_code} .*after 1 attempt$'
    with pytest.raises(RuntimeError, match=match):
        _get_pubchem_json('url')
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize('payload', [None, {}, [1, 2, 3], '{"Annot'])
def test_get_pubchem_json_retries_malformed_body(monkeypatch, payload):
    # A `2xx` without a JSON object body is a truncated/garbled transfer:
    calls, sleeps = _mock_pubchem(monkeypatch, [
        _Response(200, payload),
        _Response(200, PAGE),
    ])
    out = _get_pubchem_json('url', delay=1.0)
    assert out == PAGE
    assert len(calls) == 2
    assert sleeps == [1.0]

    calls, _ = _mock_pubchem(monkeypatch, [_Response(200, payload)] * 2)
    with pytest.raises(RuntimeError, match='HTTP 200 .*malformed body.*'
                       '2 attempts'):
        _get_pubchem_json('url', num_retries=1)
    assert len(calls) == 2


@pytest.mark.parametrize('payload', [None, [1, 2], 'not json', {'Fault': 'x'}])
def test_get_pubchem_json_non_dict_error_body(monkeypatch, payload):
    calls, _ = _mock_pubchem(monkeypatch, [_Response(503, payload)] * 2)
    with pytest.raises(RuntimeError, match='HTTP 503 .*2 attempts'):
        _get_pubchem_json('url', num_retries=1)
    assert len(calls) == 2


def test_download_redoes_step_01_after_failure(monkeypatch, tmp_path):
    dataset = MoleculeGPTDataset.__new__(MoleculeGPTDataset)
    dataset.root = str(tmp_path)
    dataset.total_page_num = 1
    dataset.total_block_num = 0

    # A run that dies on PubChem must not leave a raw dir that skips Step 01:
    _mock_pubchem(monkeypatch, [_Response(503, BUSY)] * 6)
    with pytest.raises(RuntimeError, match='PUGVIEW.ServerBusy'):
        dataset.download()
    assert not os.path.exists(f'{dataset.raw_dir}/CID2text.json')
    step1_folder = f'{dataset.raw_dir}/step_01_PubChemSTM_description'
    page_file = f'{step1_folder}/Compound_description_1.txt'
    assert not os.path.exists(page_file)

    # A page file left behind by a previous, larger run must not survive the
    # redo next to a `CID2text.json` that does not contain it:
    stale_file = f'{step1_folder}/Compound_description_7.txt'
    with open(stale_file, 'w') as f:
        f.write('7\nstale\n\n')

    # U+03B3 (Greek gamma) is not encodable in cp1252, the Windows default:
    text = 'This molecule is wet (γ-form).'
    value = {'StringWithMarkup': [{'String': 'Water is wet (γ-form).'}]}
    record = {
        'LinkedRecords': {
            'CID': [1]
        },
        'Name': 'Water',
        'Data': [{
            'Value': value
        }],
    }
    page = {'Annotations': {'Page': 1, 'Annotation': [record]}}
    calls, _ = _mock_pubchem(monkeypatch, [_Response(200, page)])
    dataset.download()
    assert len(calls) == 1
    with open(f'{dataset.raw_dir}/CID2text.json') as f:
        assert json.load(f) == {'1': [text]}
    with open(page_file, encoding='utf-8') as f:
        assert f.read() == f'1\n{text}\n\n'
    assert os.listdir(step1_folder) == ['Compound_description_1.txt']


def test_download_writes_step_01_resume_key_atomically(monkeypatch, tmp_path):
    dataset = MoleculeGPTDataset.__new__(MoleculeGPTDataset)
    dataset.root = str(tmp_path)
    dataset.total_page_num = 1
    dataset.total_block_num = 0

    # A run killed while `CID2text.json` is being written (Ctrl-C, disk full)
    # must not leave a partial file behind, since its mere existence skips
    # Step 01 on the next call, nor the temporary file it was written to:
    json_dump = json.dump

    def dump(obj, fp, **kwargs):
        if os.path.basename(fp.name).startswith('CID2text.json'):
            fp.write('{"1": ["Water is w')
            raise OSError('No space left on device')
        json_dump(obj, fp, **kwargs)

    monkeypatch.setattr(json, 'dump', dump)
    _mock_pubchem(monkeypatch, [_Response(200, PAGE)])
    with pytest.raises(OSError, match='No space left'):
        dataset.download()
    assert not os.path.exists(f'{dataset.raw_dir}/CID2text.json')
    assert not os.path.exists(f'{dataset.raw_dir}/CID2text.json.tmp')

    monkeypatch.setattr(json, 'dump', json_dump)
    calls, _ = _mock_pubchem(monkeypatch, [_Response(200, PAGE)])
    dataset.download()
    assert len(calls) == 1
    with open(f'{dataset.raw_dir}/CID2text.json') as f:
        assert json.load(f) == {}
    assert sorted(os.listdir(dataset.raw_dir)) == [
        'CID2name.json',
        'CID2name_raw.json',
        'CID2text.json',
        'CID2text_raw.json',
        'step_01_PubChemSTM_description',
    ]


def test_download_stops_at_last_pubchem_page(monkeypatch, tmp_path):
    dataset = MoleculeGPTDataset.__new__(MoleculeGPTDataset)
    dataset.root = str(tmp_path)
    dataset.total_page_num = 3
    dataset.total_block_num = 0

    # Requesting a page past `TotalPages` is a permanent PubChem HTTP 500:
    pages = [{
        'Annotations': {
            'Page': i,
            'TotalPages': 2,
            'Annotation': []
        }
    } for i in (1, 2)]
    calls, sleeps = _mock_pubchem(monkeypatch,
                                  [_Response(200, page) for page in pages])
    with pytest.warns(UserWarning, match="'total_page_num=3' exceeds the 2"):
        dataset.download()
    assert len(calls) == 2
    assert sleeps == []
    assert os.path.exists(f'{dataset.raw_dir}/CID2text.json')

    # Under a warnings-as-errors filter the warning must not discard the
    # pages just downloaded, so it is emitted after Step 01 is written:
    dataset.root = str(tmp_path / 'strict')
    calls, _ = _mock_pubchem(monkeypatch,
                             [_Response(200, page) for page in pages])
    with warnings.catch_warnings(), pytest.raises(UserWarning):
        warnings.simplefilter('error')
        dataset.download()
    assert len(calls) == 2
    assert os.path.exists(f'{dataset.raw_dir}/CID2text.json')


@pytest.mark.dataset
@withPackage('transformers', 'sentencepiece', 'accelerate', 'rdkit')
def test_molecule_gpt_dataset():
    dataset = MoleculeGPTDataset(
        root='./data/MoleculeGPT',
        num_units=10,
    )
    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 6
