import ssl

from torch_geometric.data.download import _get_ssl_context, download_url


def test_download_uses_verified_tls_by_default():
    context = _get_ssl_context()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_download_verify_ssl_opt_out(monkeypatch):
    monkeypatch.setenv('TORCH_GEOMETRIC_VERIFY_SSL', '0')
    context = _get_ssl_context()
    assert context.verify_mode == ssl.CERT_NONE


def test_download_url_passes_verified_context(monkeypatch, tmp_path):
    captured = {}

    class _FakeResponse:
        def read(self, n=-1):
            return b''

    def _fake_urlopen(url, context=None, timeout=None):
        captured['context'] = context
        return _FakeResponse()

    monkeypatch.setattr('torch_geometric.data.download.urllib.request.urlopen',
                        _fake_urlopen)

    download_url('https://example.com/data/cora.tgz', str(tmp_path), log=False)

    assert captured['context'].verify_mode == ssl.CERT_REQUIRED
    assert captured['context'].check_hostname is True


def test_download_url_passes_unverified_context_when_opt_out(
        monkeypatch, tmp_path):
    captured = {}

    class _FakeResponse:
        def read(self, n=-1):
            return b''

    def _fake_urlopen(url, context=None, timeout=None):
        captured['context'] = context
        return _FakeResponse()

    monkeypatch.setenv('TORCH_GEOMETRIC_VERIFY_SSL', '0')
    monkeypatch.setattr('torch_geometric.data.download.urllib.request.urlopen',
                        _fake_urlopen)

    download_url('https://example.com/data/cora.tgz', str(tmp_path), log=False)

    assert captured['context'].verify_mode == ssl.CERT_NONE
