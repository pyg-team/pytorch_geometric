import json
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from torch_geometric.datasets import MoleculeGPTDataset
from torch_geometric.testing import withPackage


@pytest.fixture
def pubchem(tmp_path):
    responses, pages = deque(), []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            page = int(parse_qs(urlparse(self.path).query)['page'][0])
            pages.append(page)
            status, body = responses.popleft()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Exercise only downloading descriptions, without SDF files or an LLM.
    dataset = object.__new__(MoleculeGPTDataset)
    dataset.root = str(tmp_path)
    dataset.total_page_num = 2
    dataset.total_block_num = 0
    dataset.description_url = (
        f'http://127.0.0.1:{server.server_port}/?page={{}}')
    try:
        yield dataset, responses, pages
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def annotation_page(page):
    value = {'StringWithMarkup': [{'String': 'Water is a liquid.'}]}
    record = dict(LinkedRecords={'CID': [page]}, Name='Water', Data=[{
        'Value':
        value
    }])
    return 200, {'Annotations': {'Page': page, 'Annotation': [record]}}


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_molecule_gpt_download_retry(pubchem, status):
    dataset, responses, pages = pubchem
    fault = {'Fault': {'Message': 'Server busy'}}
    responses.extend([(status, fault), annotation_page(1), annotation_page(2)])
    dataset.download()
    assert pages == [1, 1, 2]
    with open(f'{dataset.raw_dir}/CID2text.json') as f:
        assert json.load(f) == {
            '1': ['This molecule is a liquid.'],
            '2': ['This molecule is a liquid.'],
        }
    dataset.download()
    assert pages == [1, 1, 2]  # Reuse completed descriptions.


@pytest.mark.parametrize(('status', 'error', 'attempts'), [
    (404, requests.HTTPError, 1),
    (503, requests.exceptions.RetryError, 3),
    (200, ValueError, 1),
])
def test_molecule_gpt_download_resume(pubchem, tmp_path, status, error,
                                      attempts):
    dataset, responses, pages = pubchem
    responses.append(annotation_page(1))
    fault = {'Fault': {'Message': 'Unavailable'}}
    responses.extend([(status, fault)] * attempts)
    with pytest.raises(error):
        dataset.download()
    assert pages == [1] + [2] * attempts
    assert not (tmp_path / 'raw' / 'CID2text.json').exists()

    responses.extend([annotation_page(1), annotation_page(2)])
    dataset.download()
    assert pages[-2:] == [1, 2]
    with open(f'{dataset.raw_dir}/CID2text.json') as f:
        data = json.load(f)
    assert data == {
        '1': ['This molecule is a liquid.'],
        '2': ['This molecule is a liquid.'],
    }


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
