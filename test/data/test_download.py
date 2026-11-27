import ssl

import pytest

from torch_geometric.data.download import INSECURE_ENV_VAR, get_ssl_context


def test_ssl_context_verifies_certificates_by_default(monkeypatch):
    monkeypatch.delenv(INSECURE_ENV_VAR, raising=False)

    context = get_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_ssl_context_can_be_opted_out_of(monkeypatch):
    monkeypatch.setenv(INSECURE_ENV_VAR, '1')

    with pytest.warns(UserWarning, match='verification disabled'):
        context = get_ssl_context()

    assert context.verify_mode == ssl.CERT_NONE
    assert not context.check_hostname


@pytest.mark.parametrize('value', ['0', 'true', 'yes', ''])
def test_ssl_context_only_opts_out_on_exactly_one(monkeypatch, value):
    monkeypatch.setenv(INSECURE_ENV_VAR, value)

    context = get_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
